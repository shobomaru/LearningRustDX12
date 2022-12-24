use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*,
    Win32::System::LibraryLoader::*, Win32::System::Threading::*,
    Win32::System::WindowsProgramming::*, Win32::UI::WindowsAndMessaging::*,
};

const WINDOW_WIDTH: u32 = 640;
const WINDOW_HEIGHT: u32 = 360;
const BUFFER_COUNT: u32 = 3;

trait D3DBase {
    fn draw(&mut self) -> Result<()>;
    fn present(&mut self) -> Result<()>;
    fn wait(&mut self) -> Result<()>;
}

const DEFAULT_RT_CLEAR_COLOR: [f32; 4] = [ 0.1, 0.2, 0.4, 1.0 ];

struct D3D {
    #[allow(dead_code)]
    dxgi_factory: IDXGIFactory2,
    #[allow(dead_code)]
    device: ID3D12Device,
    rtv_stride: usize,
    cmd_alloc: [ID3D12CommandAllocator; BUFFER_COUNT as usize],
    cmd_queue: ID3D12CommandQueue,
    swap_chain: IDXGISwapChain3,
    cmd_list: ID3D12GraphicsCommandList,
    frame_count: u64,
    fence: ID3D12Fence,
    swap_chain_tex: [ID3D12Resource; BUFFER_COUNT as usize],
    swap_chain_heap: ID3D12DescriptorHeap,
}

impl Drop for D3D {
    fn drop(&mut self) {
        self.frame_count += 1;
        // Wait until GPU command completion
        unsafe {
            self.cmd_queue.Signal(&self.fence, self.frame_count).unwrap();
            loop {
                if self.fence.GetCompletedValue() >= self.frame_count {
                    break;
                }
            }
        }
    }
}

impl D3D {
    fn new(width: u32, height: u32, hwnd: HWND) -> Self {
        let factory_flags = if cfg!(debug_assertions) { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(factory_flags) }.unwrap();

        if cfg!(debug_assertions) {
            let mut debug: Option<ID3D12Debug> = None;
            unsafe {
                match D3D12GetDebugInterface(&mut debug) {
                    Ok(_) => { debug.unwrap().EnableDebugLayer(); println!("Enable debug"); },
                    _ => {},
                }
            }
            let mut debug: Option<ID3D12Debug1> = None;
            unsafe {
                match D3D12GetDebugInterface(&mut debug) {
                    Ok(_) => { debug.unwrap().SetEnableGPUBasedValidation(BOOL(1));
                                println!("Enable GPU based validation"); },
                    _ => {},
                }
            }
        }
        let device: ID3D12Device = {
            let mut device_ptr: Option<ID3D12Device> = None;
            unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device_ptr) }.unwrap();
            device_ptr.unwrap()
        };
        let rtv_stride = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) } as usize;

        let cmd_alloc: [_; BUFFER_COUNT as usize] =
            array_init::try_array_init(|_: usize| -> Result<ID3D12CommandAllocator> {
                let r = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
                Ok(r)
            }).unwrap();
        let cmd_queue: ID3D12CommandQueue = unsafe {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default()
            };
            device.CreateCommandQueue(&desc)
        }.unwrap();

        let swap_chain : IDXGISwapChain3 = unsafe {
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                BufferCount: BUFFER_COUNT,
                Width: width as u32,
                Height: height as u32,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                BufferUsage: DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            factory.CreateSwapChainForHwnd(&cmd_queue, hwnd, &desc, std::ptr::null(), None)
        }.unwrap().cast().unwrap();
        
        let cmd_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_alloc[0], None)
        }.unwrap();
        unsafe { cmd_list.Close() }.unwrap();

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();

        let swap_chain_heap: ID3D12DescriptorHeap = unsafe {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 10,
                ..Default::default()
            };
            device.CreateDescriptorHeap(&desc)
        }.unwrap();
        let swap_chain_tex: [_; BUFFER_COUNT as usize] =
            array_init::try_array_init(|i| -> Result<ID3D12Resource> {
                let r = unsafe { swap_chain.GetBuffer(i as u32) }?;
                Ok(r)
            }).unwrap();
        let h_rtv = unsafe { swap_chain_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in swap_chain_tex.iter().enumerate() {
            let desc = D3D12_CPU_DESCRIPTOR_HANDLE{
                ptr: h_rtv.ptr + i.0 * rtv_stride
            };
            unsafe { device.CreateRenderTargetView(i.1, std::ptr::null(), &desc) };
        }

        D3D{
            dxgi_factory: factory,
            device,
            rtv_stride,
            cmd_alloc,
            cmd_queue,
            swap_chain,
            cmd_list,
            frame_count: 0,
            fence,
            swap_chain_tex,
            swap_chain_heap,
        }
    }
}

impl D3DBase for D3D {
    fn draw(&mut self) -> Result<()> {
        self.frame_count += 1;
        let frame_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() };

        let cmd_alloc = &self.cmd_alloc[self.frame_count as usize % 3];
        unsafe { cmd_alloc.Reset() }?;
        unsafe { self.cmd_list.Reset(cmd_alloc, None) }?;

        let before_swapchain_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: Some(self.swap_chain_tex[frame_index as usize].clone()),
                    StateBefore: D3D12_RESOURCE_STATE_COMMON,
                    StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
        };
        unsafe { self.cmd_list.ResourceBarrier(&[before_swapchain_barrier]) };

        let h_rtv = unsafe { self.swap_chain_heap.GetCPUDescriptorHandleForHeapStart() };
        let rtv_swapchain = D3D12_CPU_DESCRIPTOR_HANDLE{
            ptr: h_rtv.ptr + frame_index as usize % 3 * self.rtv_stride as usize
        };
        unsafe { self.cmd_list.ClearRenderTargetView(rtv_swapchain, DEFAULT_RT_CLEAR_COLOR.as_ptr(), &[]) };

        let after_swapchain_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: Some(self.swap_chain_tex[frame_index as usize].clone()),
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COMMON,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
        };
        unsafe { self.cmd_list.ResourceBarrier(&[after_swapchain_barrier]) };

        unsafe { self.cmd_list.Close() }?;

        let cmds = [Some(ID3D12CommandList::from(&self.cmd_list))];
        unsafe { self.cmd_queue.ExecuteCommandLists(&cmds) };
        unsafe { self.cmd_queue.Signal(&self.fence, self.frame_count) }?;

        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        let param: DXGI_PRESENT_PARAMETERS = Default::default();
        unsafe { self.swap_chain.Present1(1, 0, &param) }?;
        Ok(())
    }

    fn wait(&mut self) -> Result<()> {
        if self.frame_count != 0 {
            unsafe { self.fence.SetEventOnCompletion(self.frame_count, None) }?;
        }
        Ok(())
    }
}

extern "system" fn wndproc(
    window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM, ) -> LRESULT {
    match message {
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT::default()
        }
        _ => {
            unsafe { DefWindowProcW(window, message, wparam, lparam) }
        }
    }
}

fn setup_window(width: u32, height: u32) -> HWND {
    let class_name = "WindowClass\0".encode_utf16().collect::<Vec<u16>>();

    let wcex = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wndproc),
        hInstance: unsafe { GetModuleHandleW(None).unwrap() },
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap() },
        lpszClassName: PCWSTR(class_name.as_ptr()),
        ..Default::default()
    };
    assert_ne!(unsafe { RegisterClassExW(&wcex) }, 0);

    let mut rect = RECT {
        left: 0, top: 0, right: width as i32, bottom: height as i32
    };
    unsafe{ AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false) };

    let window_width = rect.right - rect.left;
    let window_height: i32 = rect.bottom - rect.top;

    let hwnd = unsafe { CreateWindowExW(
        Default::default(),
        PCWSTR(class_name.as_ptr()),
        PCWSTR("Window\0".encode_utf16().collect::<Vec<u16>>().as_ptr()),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, window_width, window_height,
        None, None, None, std::ptr::null()
    ) };
    assert_ne!(hwnd.0, 0);

    unsafe { ShowWindow(hwnd, SW_SHOW) };

    hwnd
}

fn main() -> Result<()> {
    
    let main_window_handle = setup_window(WINDOW_WIDTH, WINDOW_HEIGHT);
    let mut d3d = D3D::new(WINDOW_WIDTH, WINDOW_HEIGHT, main_window_handle);

    let mut msg = MSG::default();
    loop {
        if msg.message == WM_QUIT {
            break;
        }
        if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
            unsafe { DispatchMessageW(&msg) };
        }
        else {
            d3d.wait().unwrap();
            d3d.draw().unwrap();
            d3d.present().unwrap();
        }
    }

    match msg.wParam.0 {
        0 => Ok(()),
        _ => panic!("wParam {}", msg.wParam.0)
    }
}
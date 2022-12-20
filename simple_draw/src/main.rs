use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*, Win32::Graphics::{Dxgi::*, Gdi::*},
    Win32::System::LibraryLoader::*, Win32::System::Threading::*,
    Win32::System::WindowsProgramming::*, Win32::UI::WindowsAndMessaging::*,
};

use std::mem::transmute;

const WINDOW_WIDTH: u32 = 640;
const WINDOW_HEIGHT: u32 = 360;
const BUFFER_COUNT: u32 = 3;

trait D3DBase {
    fn draw(&mut self) {}
    fn present(&mut self) {}
    fn wait(&mut self) {}
}

struct D3D {
    dxgi_factory: IDXGIFactory2,
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
    fn new(width: u32, height: u32, hwnd: HWND) -> Result<Self> {
        let factory: IDXGIFactory2 = unsafe { CreateDXGIFactory2(0) }?;
        let device: ID3D12Device = {
            let mut device_ptr: Option<ID3D12Device> = None;
            unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device_ptr) }?;
            device_ptr.unwrap()
        };
        let rtv_stride = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) } as usize;
        let cmd_alloc: [_; BUFFER_COUNT as usize] =
            array_init::try_array_init(|_: usize| -> Result<ID3D12CommandAllocator> {
                let r = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
                Ok(r)
            })?;
        let cmd_queue: ID3D12CommandQueue = unsafe {
            let desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default()
            };
            device.CreateCommandQueue(&desc)
        }?;
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
        }?.cast()?;
        let cmd_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_alloc[0], None)
        }?;
        unsafe { cmd_list.Close()? };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
        let swap_chain_heap: ID3D12DescriptorHeap = unsafe {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 10,
                ..Default::default()
            };
            device.CreateDescriptorHeap(&desc)
        }?;
        let h_rtv = unsafe { swap_chain_heap.GetCPUDescriptorHandleForHeapStart() };
        let swap_chain_tex: [_; BUFFER_COUNT as usize] =
            array_init::try_array_init(|i| -> Result<ID3D12Resource> {
                let r = unsafe { swap_chain.GetBuffer(i as u32) }?;
                Ok(r)
            })?;
        for i in swap_chain_tex.iter().enumerate() {
            let desc = D3D12_CPU_DESCRIPTOR_HANDLE{
                ptr: h_rtv.ptr + i.0 * rtv_stride
            };
            unsafe { device.CreateRenderTargetView(i.1, std::ptr::null(), &desc) };
        }
        Ok(D3D{
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
        })
    }
}

impl D3DBase for D3D {
    fn draw(&mut self) {
        //
    }

    fn present(&mut self) {
        unsafe { self.swap_chain.Present(1, 0).unwrap() };
    }

    fn wait(&mut self) {
        //
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
    let wcex = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wndproc),
        hInstance: unsafe { GetModuleHandleW(None).unwrap() },
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap() },
        lpszClassName: PCWSTR("WindowClass\0".encode_utf16().collect::<Vec<u16>>().as_ptr()),
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
        PCWSTR("WindowClass\0".encode_utf16().collect::<Vec<u16>>().as_ptr()),
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
    let mut d3d = D3D::new(WINDOW_WIDTH, WINDOW_HEIGHT, main_window_handle)?;

    let mut msg = MSG::default();
    loop {
        if msg.message == WM_QUIT {
            break;
        }
        if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
            unsafe { DispatchMessageW(&msg) };
        }
        else {
            d3d.wait();
            d3d.draw();
            d3d.present();
        }
    }

    match msg.wParam.0 {
        0 => Ok(()),
        _ => panic!("wParam {}", msg.wParam.0)
    }
}

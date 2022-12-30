// Dxc doesn't use in this sample
#[allow(unused_imports)]
use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Dxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*,
    Win32::System::LibraryLoader::*, Win32::System::Threading::*,
    Win32::UI::WindowsAndMessaging::*,
    Win32::Security::*, Win32::System::Memory::*,
};
use std::sync::{Arc, atomic::AtomicUsize};
use libc::{c_uint, c_char};

const BUFFER_COUNT: u32 = 3;

#[macro_export]
macro_rules! align {
    ($val:expr, $align:expr) => {{
        let a = $val as usize;
        let b = $align as usize;
        (a + b - 1) & !(b - 1)
    }}
}

pub trait D3DBase {
    fn draw(&mut self);
    fn present(&mut self) -> Result<()>;
    fn wait(&mut self) -> Result<()>;
    fn get_image(&mut self) -> (&ID3D12CommandQueue, &ID3D12Resource);
}

pub fn catch_up_d3d_log(log_atomic: Arc::<AtomicUsize>) -> std::thread::JoinHandle::<()>
{
    // VSCode doesn't handle OutputDebugString() which is used by D3D debug layer
    // so we manually read and print the strings

    let sd = SECURITY_DESCRIPTOR{ ..Default::default() };
    let sa = SECURITY_ATTRIBUTES{
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: &sd as *const _ as *mut _,
        bInheritHandle: BOOL(1)
    };
    unsafe { InitializeSecurityDescriptor(PSECURITY_DESCRIPTOR(&sd as *const _ as *mut _), 1) };
    unsafe { SetSecurityDescriptorDacl(PSECURITY_DESCRIPTOR(&sd as *const _ as *mut _), BOOL(1), None, BOOL(0)) };

    let db_ack = "DBWIN_BUFFER_READY\0".encode_utf16().collect::<Vec<u16>>();
    let db_rdy = "DBWIN_DATA_READY\0".encode_utf16().collect::<Vec<u16>>();
    let h_ack = unsafe { CreateEventW(Some(&sa), BOOL(0), BOOL(0), PCWSTR(db_ack.as_ptr())) }.unwrap();
    let h_rdy = unsafe { CreateEventW(Some(&sa), BOOL(0), BOOL(0), PCWSTR(db_rdy.as_ptr())) }.unwrap();

    let log_size = 8192u32;
    let db = "DBWIN_BUFFER\0".encode_utf16().collect::<Vec<u16>>();
    let fh = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, Some(&sa), PAGE_READWRITE, 0, log_size, PCWSTR(db.as_ptr())) }.unwrap();

    let pid = unsafe { GetCurrentProcessId() };

    let thread = std::thread::spawn(move || {
        let mmf = unsafe { MapViewOfFile(fh, FILE_MAP_READ, 0, 0, log_size as usize) };
        loop {
            unsafe { SetEvent(h_ack) };
            let wait = unsafe { WaitForSingleObject(h_rdy, 100) };

            if log_atomic.load(std::sync::atomic::Ordering::Acquire) != 0 {
                break;
            }
            if wait == WAIT_OBJECT_0 {
                let log_pid = unsafe { *(mmf as *const c_uint) };
                if pid == log_pid {
                    let log_ptr = (mmf as isize + 4) as *const c_char;
                    let log_msg = unsafe { std::ffi::CStr::from_ptr(log_ptr) };
                    println!("MSG - {}", log_msg.to_str().unwrap());
                }
            }
        }
        unsafe {
            UnmapViewOfFile(mmf);
            CloseHandle(fh);
            CloseHandle(h_ack);
            CloseHandle(h_rdy);
        }
        println!("Log thread finished");
    });
    thread
}

const DEFAULT_RT_CLEAR_COLOR: [f32; 4] = [ 0.1, 0.2, 0.4, 1.0 ];

pub struct D3D {
    pub dxgi_factory: IDXGIFactory5,
    pub device: ID3D12Device,
    rtv_stride: usize,
    cmd_alloc: [ID3D12CommandAllocator; BUFFER_COUNT as usize],
    cmd_queue: ID3D12CommandQueue,
    swap_chain: IDXGISwapChain3,
    cmd_list: ID3D12GraphicsCommandList,
    frame_count: u64,
    fence: ID3D12Fence,
    swap_chain_tex: Option<[ID3D12Resource; BUFFER_COUNT as usize]>,
    swap_chain_heap: ID3D12DescriptorHeap,
    is_fullscreen: bool,
}

impl Drop for D3D {
    fn drop(&mut self) {
        self.frame_count += 1;
        // Wait for GPU command completion
        unsafe {
            self.cmd_queue.Signal(&self.fence, self.frame_count).unwrap();
            self.fence.SetEventOnCompletion(self.frame_count, None).unwrap();
            self.device.GetDeviceRemovedReason().unwrap();
        }
        // Release fullscreen state
        let mut is_fullscreen = BOOL(0);
        unsafe { self.swap_chain.GetFullscreenState(Some(&mut is_fullscreen), None) }.unwrap();
        if is_fullscreen.into() {
            unsafe { self.swap_chain.SetFullscreenState(BOOL(0), None) }.unwrap();
        }
    }
}

impl D3D {
    pub fn new(width: u32, height: u32, hwnd: HWND, is_sw: bool) -> Self {

        let factory_flags = if cfg!(debug_assertions) { DXGI_CREATE_FACTORY_DEBUG } else { 0 };
        let factory: IDXGIFactory5 = unsafe { CreateDXGIFactory2(factory_flags) }.unwrap();

        if cfg!(debug_assertions) {
            let mut debug: Option<ID3D12Debug> = None;
            unsafe {
                match D3D12GetDebugInterface(&mut debug) {
                    Ok(_) => {
                        debug.as_ref().unwrap().EnableDebugLayer();
                        println!("Enable debug");
                    },
                    _ => { println!("Cannot enable debug layer. Maybe developer mode is disabled.") },
                }
            }
            let mut debug: Option<ID3D12Debug1> = None;
            unsafe {
                match D3D12GetDebugInterface(&mut debug) {
                    Ok(_) => {
                        debug.as_ref().unwrap().SetEnableGPUBasedValidation(BOOL(1));
                        debug.as_ref().unwrap().SetEnableSynchronizedCommandQueueValidation(BOOL(1));
                        println!("Enable GPU based validation");
                    },
                    _ => { println!("Cannot get ID3D12Debug1 interface.") },
                }
            }
        }

        let mut allow_tearing = BOOL(0);
        if let Ok(_) = unsafe { factory.CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, std::ptr::addr_of_mut!(allow_tearing) as *mut _, 4) } {
            println!("VRR support : {:?}", allow_tearing);
        }

        let adapter: IDXGIAdapter4 = {
            let mut adapter: Option<IDXGIAdapter4> = None;
            if is_sw {
                adapter = unsafe { factory.EnumWarpAdapter() }.ok();
            }
            if adapter.is_none() {
                adapter = unsafe { factory.EnumAdapters1(0).and_then(|a| a.cast::<IDXGIAdapter4>()) }.ok();
            }
            adapter.unwrap()
        };
        let adapter_desc = unsafe { adapter.GetDesc3() }.unwrap();
        println!("Adapter name: {}", String::from_utf16_lossy(adapter_desc.Description.split(|n| n == &0).next().unwrap()).to_string());
        let device: ID3D12Device = {
            let mut device_ptr: Option<ID3D12Device> = None;
            unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device_ptr) }.unwrap();
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
                BufferUsage: DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_SHADER_INPUT,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            factory.CreateSwapChainForHwnd(&cmd_queue, hwnd, &desc, None, None)
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
            unsafe { device.CreateRenderTargetView(i.1, None, desc) };
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
            swap_chain_tex: Some(swap_chain_tex),
            swap_chain_heap,
            is_fullscreen: false,
        }
    }
}

impl D3DBase for D3D {
    fn draw(&mut self) {
        self.frame_count += 1;
        let frame_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() };

        let cmd_alloc = &self.cmd_alloc[self.frame_count as usize % 3];
        unsafe { cmd_alloc.Reset() }.unwrap();
        unsafe { self.cmd_list.Reset(cmd_alloc, None) }.unwrap();
        
        let swap_chain_tex = &((self.swap_chain_tex.as_ref().unwrap())[frame_index as usize]);
        let swap_chain_copy = swap_chain_tex.clone();
        let before_swapchain_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // I don't know why pResource reqires non-referenced type. This will make a resource leak :(
                    pResource: Some(unsafe { std::mem::transmute_copy(&swap_chain_copy) }),
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
                    pResource: Some(unsafe { std::mem::transmute_copy(&swap_chain_copy) }),
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COMMON,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
        };
        unsafe { self.cmd_list.ResourceBarrier(&[after_swapchain_barrier]) };

        unsafe { self.cmd_list.Close() }.unwrap();

        let cmds = [Some(ID3D12CommandList::from(&self.cmd_list))];
        unsafe { self.cmd_queue.ExecuteCommandLists(&cmds) };
        unsafe { self.cmd_queue.Signal(&self.fence, self.frame_count) }.unwrap();
    }

    fn present(&mut self) -> Result<()> {
        let param: DXGI_PRESENT_PARAMETERS = Default::default();
        unsafe { self.swap_chain.Present1(1, 0, &param) }.unwrap();
        Ok(())
    }

    fn wait(&mut self) -> Result<()> {
        if self.frame_count != 0 {
            unsafe { self.fence.SetEventOnCompletion(self.frame_count - 1, None) }?;
        }
        // Handling fullscreen state
        let mut new_fullscreen = BOOL(0);
        unsafe { self.swap_chain.GetFullscreenState(Some(&mut new_fullscreen), None) }?;
        if new_fullscreen != self.is_fullscreen {
            println!("Window state changed. fullscreen={:?}", new_fullscreen);
            self.frame_count += 1;
            unsafe { self.cmd_queue.Signal(&self.fence, self.frame_count) }?;
            unsafe { self.fence.SetEventOnCompletion(self.frame_count, None) }?;
            self.swap_chain_tex = None;

            let desc = unsafe { self.swap_chain.GetDesc1() }?;
            unsafe { self.swap_chain.ResizeBuffers(
                desc.BufferCount, desc.Width, desc.Height, desc.Format, 0)
            }?;
            let swap_chain_tex: [_; BUFFER_COUNT as usize] =
                array_init::try_array_init(|i| -> Result<ID3D12Resource> {
                    let r = unsafe { self.swap_chain.GetBuffer(i as u32) }?;
                    Ok(r)
                }).unwrap();
            let h_rtv = unsafe { self.swap_chain_heap.GetCPUDescriptorHandleForHeapStart() };
            for i in swap_chain_tex.iter().enumerate() {
                let desc = D3D12_CPU_DESCRIPTOR_HANDLE{
                    ptr: h_rtv.ptr + i.0 * self.rtv_stride
                };
                unsafe { self.device.CreateRenderTargetView(i.1, None, desc) };
            }
            self.swap_chain_tex = Some(swap_chain_tex);
            
            self.is_fullscreen = new_fullscreen.into();
        }
        Ok(())
    }
    
    fn get_image(&mut self) -> (&ID3D12CommandQueue, &ID3D12Resource)
    {
        let r = self.swap_chain_tex.as_ref().unwrap();
        (&self.cmd_queue, &r[0])
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

pub fn setup_window(width: u32, height: u32) -> HWND {
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
        None, None, None, None
    ) };
    assert_ne!(hwnd.0, 0);

    unsafe { ShowWindow(hwnd, SW_SHOW) };

    hwnd
}

pub fn default_main(width: u32, height: u32) -> Result<()> {
    use crate::simple_draw::*;

    let dbg_atomic = Arc::new(AtomicUsize::new(0));
    let mut dbg_thread: Option<std::thread::JoinHandle<()>> = None;
    if cfg!(debug_assertions) {
        dbg_thread = Some(catch_up_d3d_log(dbg_atomic.clone()));
    }

    let mut debug_device: Option<ID3D12DebugDevice> = None;
    let mut msg = MSG::default();
    {
        let main_window_handle = setup_window(width, height);
        let mut d3d = D3D::new(width, height, main_window_handle, false);
        if cfg!(debug_assertions) {
            debug_device = d3d.device.cast::<ID3D12DebugDevice>().ok();
        }
        
        loop {
            if msg.message == WM_QUIT {
                break;
            }
            if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
                unsafe { DispatchMessageW(&msg) };
            }
            else {
                d3d.wait().unwrap();
                d3d.draw();
                d3d.present().unwrap();
            }
        }
    }
    unsafe {
        if let Some(d) = debug_device {
            // Expect that only a ID3D12Device leaks
            d.ReportLiveDeviceObjects(D3D12_RLDO_IGNORE_INTERNAL).unwrap();
        }
    }

    if let Some(x) = dbg_thread {
        // Exit log thread
        dbg_atomic.store(1, std::sync::atomic::Ordering::Release);
        x.join().unwrap();
    }

    match msg.wParam.0 {
        0 => Ok(()),
        _ => panic!("wParam {}", msg.wParam.0)
    }
}

use windows::{
    core::*, Win32::Foundation::*,
    Win32::System::LibraryLoader::*,
    Win32::UI::WindowsAndMessaging::*,
};

static WINDOW_WIDTH: u32 = 640;
static WINDOW_HEIGHT: u32 = 360;

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
        None, None, None, None
    ) };
    assert_ne!(hwnd.0, 0);

    unsafe { ShowWindow(hwnd, SW_SHOW) };

    hwnd
}

fn main() -> Result<()> {
    
    let _main_window_handle = setup_window(WINDOW_WIDTH, WINDOW_HEIGHT);

    let mut msg = MSG::default();
    loop {
        if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
            unsafe { DispatchMessageW(&msg) };
        }
        if msg.message == WM_QUIT {
            break;
        }
    }

    match msg.wParam.0 {
        0 => Ok(()),
        _ => panic!("wParam {}", msg.wParam.0)
    }
}

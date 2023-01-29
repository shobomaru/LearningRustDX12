pub mod simple_texture;

#[cfg(test)]
mod tests {

    const WINDOW_WIDTH: u32 = 640;
    const WINDOW_HEIGHT: u32 = 360;

    use windows::{
        Win32::Graphics::Direct3D12::*, Win32::UI::WindowsAndMessaging::*,
        Win32::Graphics::Dxgi::Common::*,
    };
    use std::sync::{Arc, atomic::AtomicUsize};
    use libc::*;
    use crypto::{digest::Digest, sha1::Sha1};
    use crate::simple_texture::*;

    #[test]
    fn test_image() {
        let expected_sha1 = "b0ec33cc7ede9b5d7e818705ffb966257f49ac26";
        let mut rendered_sha1 = String::new();

        let dbg_atomic = Arc::new(AtomicUsize::new(0));
        let mut dbg_thread: Option<std::thread::JoinHandle<()>> = None;
        if cfg!(debug_assertions) {
           dbg_thread = Some(catch_up_d3d_log(dbg_atomic.clone()));
        }

        let mut msg = MSG::default();
        {
            let main_window_handle = setup_window(WINDOW_WIDTH, WINDOW_HEIGHT);
            let mut d3d = D3D::new(WINDOW_WIDTH, WINDOW_HEIGHT, main_window_handle, true);
            
            loop {
                if msg.message == WM_QUIT {
                    break;
                }
                if unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.into() {
                    unsafe { DispatchMessageW(&msg) };
                }
                else {
                    for _ in 0..20 {
                        d3d.wait().unwrap();
                        d3d.draw();
                        d3d.present().unwrap();
                    }
                    let img = d3d.get_image();
                    rendered_sha1 = hash_image(img.0, img.1);
                    unsafe { PostQuitMessage(0) };
                }
            }
        }
        if let Some(x) = dbg_thread {
            dbg_atomic.store(1, std::sync::atomic::Ordering::Release);
            x.join().unwrap();
        }

        assert_eq!(expected_sha1, rendered_sha1);
    }

    fn hash_image(cmd_queue: &ID3D12CommandQueue, image: &ID3D12Resource) -> String {
        let mut device = None as Option<ID3D12Device>;
        unsafe { cmd_queue.GetDevice(&mut device) }.unwrap();
        let device = device.unwrap();
        let res_desc = unsafe { image.GetDesc() };
        assert_eq!(res_desc.DepthOrArraySize, 1);
        assert_eq!(res_desc.MipLevels, 1);
        assert_eq!(res_desc.Format, DXGI_FORMAT_R8G8B8A8_UNORM);
        let mut res: Option<ID3D12Resource> = None;
        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT{ ..Default::default() };
        let mut row_pitch = 0;
        {
            // Create a readable resource
            let mut desc = res_desc;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut heap_prop = D3D12_HEAP_PROPERTIES{ ..Default::default() };
            let mut total_size = 0u64;
            unsafe { device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut layout), None, Some(&mut row_pitch), Some(&mut total_size)) };
            assert_ne!(total_size, 0);
            assert_eq!(row_pitch as usize, crate::align!(4 * res_desc.Width, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Width = total_size;
            desc.Height = 1;
            heap_prop.Type = D3D12_HEAP_TYPE_READBACK;
            unsafe { device.CreateCommittedResource(&heap_prop, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut res) }.unwrap();
        }
        let res = &res.unwrap();
        {
            // Copy from an image to a readable reource
            let cmd_alloc = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();
            let cmd_list: ID3D12GraphicsCommandList = unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_alloc, None) }.unwrap();
            let image_copy = image.clone();
            let before_swapchain_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: Some(unsafe { std::mem::transmute_copy(&image_copy) }),
                        StateBefore: D3D12_RESOURCE_STATE_COMMON,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    }),
                },
            };
            unsafe { cmd_list.ResourceBarrier(&[before_swapchain_barrier]) };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                // pResource causes unnecessary adding reference counter, but it works
                pResource: Some(image.clone()),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0
                }
            };
            let dest = D3D12_TEXTURE_COPY_LOCATION {
                pResource: Some(res.clone()),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: res_desc.Format,
                            Width: res_desc.Width as u32,
                            Height: res_desc.Height,
                            Depth: 1,
                            RowPitch: row_pitch as u32,
                        },
                        Offset: 0,
                    }
                }
            };
            unsafe { cmd_list.CopyTextureRegion(&dest, 0, 0, 0, &src, None) };
            unsafe { cmd_list.Close() }.unwrap();
            let cmds = [Some(ID3D12CommandList::from(&cmd_list))];
            unsafe { cmd_queue.ExecuteCommandLists(&cmds) };
            let fence: ID3D12Fence = unsafe{ device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
            unsafe { cmd_queue.Signal(&fence, 1) }.unwrap();
            unsafe { fence.SetEventOnCompletion(1, None) }.unwrap();
        }
        let mut sha1 = Sha1::new();
        {
            let mut p = 0 as *mut c_void;
            unsafe { res.Map(0, None, Some(&mut p)) }.unwrap();
            for y in 0..res_desc.Height as isize {
                for x in 0..res_desc.Width as isize {
                    let pt = (p as isize + y * row_pitch as isize + x * 4) as *const c_uint;
                    let s: [u8; 4] = unsafe { *pt }.to_le_bytes();
                    sha1.input(&s);
                }
            }
            unsafe { res.Unmap(0, None) };
        }
        sha1.result_str()
    }

}

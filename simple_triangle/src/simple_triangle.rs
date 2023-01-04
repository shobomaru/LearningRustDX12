use windows::{
    core::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Dxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D12::*, Win32::Graphics::Dxgi::Common::*, Win32::Graphics::Dxgi::*,
    Win32::System::LibraryLoader::*, Win32::System::Threading::*,
    Win32::UI::WindowsAndMessaging::*,
    Win32::Security::*, Win32::System::Memory::*,
};
use std::sync::{Arc, atomic::AtomicUsize};
use std::ffi::CString;
use libc::{c_uint, c_char, c_void};
use directx_math::*;
use std::f32::consts::PI;

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
    #[allow(dead_code)]
    rtv_stride: usize,
    #[allow(dead_code)]
    dsv_stride: usize,
    #[allow(dead_code)]
    res_stride: usize,
    #[allow(dead_code)]
    sampler_stride: usize,
    cmd_alloc: [ID3D12CommandAllocator; BUFFER_COUNT as usize],
    cmd_queue: ID3D12CommandQueue,
    swap_chain: IDXGISwapChain3,
    cmd_list: ID3D12GraphicsCommandList,
    frame_count: u64,
    fence: ID3D12Fence,
    swap_chain_tex: Option<[ID3D12Resource; BUFFER_COUNT as usize]>,
    swap_chain_heap: ID3D12DescriptorHeap,
    is_fullscreen: bool,
    scene: Scene,
    resource: Resource,
}

struct Scene {
    camera_pos: XMVECTOR,
    camera_target: XMVECTOR,
    camera_up: XMVECTOR,
    camera_fov: f32,
    camera_near: f32,
    camera_far: f32,
}

struct Resource {
    width: u32,
    height: u32,
    rootsig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    #[allow(dead_code)]
    vb_ib: ID3D12Resource,
    vb_view: D3D12_VERTEX_BUFFER_VIEW,
    ib_view: D3D12_INDEX_BUFFER_VIEW,
    ib_count: u32,
    cb: [ID3D12Resource; BUFFER_COUNT as usize],
}

const SPHERE_STACKS: u32 = 8;
const SPHERE_SLICES: u32 = 8;

fn create_resources(device: &ID3D12Device, width: u32, height: u32) -> Resource {
    // Create a root signature
    let rootsig: ID3D12RootSignature = {
        let root_param = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0, RegisterSpace: 0
                }
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        };
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &root_param,
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        let mut root_sig_blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        unsafe { D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut root_sig_blob, Some(&mut err)) }.unwrap();
        let root_sig_blob = root_sig_blob.unwrap();
        let ary = unsafe { std::ptr::slice_from_raw_parts(root_sig_blob.GetBufferPointer() as *const u8, root_sig_blob.GetBufferSize()) };
        unsafe { device.CreateRootSignature(0, &*ary) }.unwrap()
    };

    // DLL load path for dxil.dll
    {
        let path = "../dll\0".encode_utf16().collect::<Vec<u16>>();
        unsafe { SetDllDirectoryW(PCWSTR(path.as_ptr())) };
    }

    // Craete a PSO
    let pso: ID3D12PipelineState;
    {
        let code_scene_vs = r#"
cbuffer CScene {
    float4x4 ViewProj;
};
struct Output {
    float4 position : SV_Position;
    float3 world : World;
    float3 normal : Normal;
};
Output main(float3 position : Position, float3 normal : Normal) {
    Output output;
    output.position = mul(float4(position, 1), ViewProj);
    output.world = position;
    output.normal = normalize(normal);
    return output;
}
"#;

        let code_scene_ps = r#"
struct Input {
    float4 position : SV_Position;
    float3 world : World;
    float3 normal : Normal;
};
float4 main(Input input) : SV_Target {
    float3 intensity = input.normal * 0.5 + 0.5;
    return float4(intensity, 1);
}
"#;

        let dxc: IDxcCompiler = unsafe { DxcCreateInstance(&CLSID_DxcCompiler) }.unwrap();
        let dxclib: IDxcLibrary = unsafe { DxcCreateInstance(&CLSID_DxcLibrary) }.unwrap();

        let txt_scene_vs = unsafe {
            dxclib.CreateBlobWithEncodingFromPinned(code_scene_vs.as_ptr() as *const c_void, code_scene_vs.len().try_into().unwrap(), DXC_CP_UTF8)
        }.unwrap();
        let txt_scene_ps = unsafe {
            dxclib.CreateBlobWithEncodingFromPinned(code_scene_ps.as_ptr() as *const c_void, code_scene_ps.len().try_into().unwrap(), DXC_CP_UTF8)
        }.unwrap();

        let mut shader_args_u16: [_; 3] = array_init::array_init(|i| {
            let s = ["-Zi\0", "-all_resources_bound\0", "-Qembed_debug\0"];
            s[i].encode_utf16().collect::<Vec<u16>>()
        });
        let shader_args_p: [_; 3] = array_init::array_init(|i| {
            PWSTR(shader_args_u16[i].as_mut_ptr())
        });
        let entry = "main\0".encode_utf16().collect::<Vec<u16>>();
        let profile_vs = "vs_6_0\0".encode_utf16().collect::<Vec<u16>>();
        let profile_ps = "ps_6_0\0".encode_utf16().collect::<Vec<u16>>();

        let res_scene_vs = unsafe {
            dxc.Compile(&txt_scene_vs, None, PCWSTR(entry.as_ptr()),
                PCWSTR(profile_vs.as_ptr()), Some(&shader_args_p), &[] as _, None)
        }.unwrap();
        if unsafe { res_scene_vs.GetStatus().unwrap() } != S_OK {
            panic!("{}", unsafe { CString::from_raw(res_scene_vs.GetErrorBuffer().unwrap().GetBufferPointer() as *mut i8).to_string_lossy().to_string() });
        }
        let bin_scene_vs = unsafe { res_scene_vs.GetResult() }.unwrap();

        let res_scene_ps = unsafe {
            dxc.Compile(&txt_scene_ps, None, PCWSTR(entry.as_ptr()),
                PCWSTR(profile_ps.as_ptr()), Some(&shader_args_p), &[] as _, None)
        }.unwrap();
        if unsafe { res_scene_ps.GetStatus().unwrap() } != S_OK {
            panic!("{}", unsafe { CString::from_raw(res_scene_ps.GetErrorBuffer().unwrap().GetBufferPointer() as *mut i8).to_string_lossy().to_string() });
        }
        let bin_scene_ps = unsafe { res_scene_ps.GetResult() }.unwrap();

        let ie_desc = [
            D3D12_INPUT_ELEMENT_DESC{
                SemanticName: PCSTR("POSITION\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC{
                SemanticName: PCSTR("NORMAL\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];
        let mut rs_desc: D3D12_RASTERIZER_DESC = unsafe { std::mem::zeroed() };
        rs_desc.CullMode = D3D12_CULL_MODE_BACK;
        rs_desc.FillMode = D3D12_FILL_MODE_SOLID;
        rs_desc.DepthClipEnable = BOOL(1);
        let mut ds_desc: D3D12_DEPTH_STENCIL_DESC = unsafe { std::mem::zeroed() };
        //ds_desc.DepthEnable = BOOL(1);
        //ds_desc.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        let mut bs_desc: D3D12_BLEND_DESC = unsafe { std::mem::zeroed() };
        bs_desc.RenderTarget[0].RenderTargetWriteMask = 0b1111;

        let mut pso_desc_scene = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: Some(rootsig.clone()),
            VS: unsafe { D3D12_SHADER_BYTECODE{
                pShaderBytecode: bin_scene_vs.GetBufferPointer(),
                BytecodeLength: bin_scene_vs.GetBufferSize()}
            },
            PS: unsafe { D3D12_SHADER_BYTECODE{
                pShaderBytecode: bin_scene_ps.GetBufferPointer(),
                BytecodeLength: bin_scene_ps.GetBufferSize()}
            },
            DS: unsafe{ std::mem::zeroed() },
            HS: unsafe{ std::mem::zeroed() },
            GS: unsafe{ std::mem::zeroed() },
            InputLayout: D3D12_INPUT_LAYOUT_DESC{
                pInputElementDescs: ie_desc.as_ptr(),
                NumElements: ie_desc.len().try_into().unwrap(),
            },
            IBStripCutValue: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            RasterizerState: rs_desc,
            DepthStencilState: ds_desc,
            BlendState: bs_desc,
            SampleMask: u32::MAX,
            NumRenderTargets: 1,
            RTVFormats: [DXGI_FORMAT_UNKNOWN; 8],
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC{ Count: 1, Quality: 0 },
            StreamOutput: unsafe{ std::mem::zeroed() },
            NodeMask: 0,
            CachedPSO: unsafe{ std::mem::zeroed() },
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        pso_desc_scene.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso = unsafe { device.CreateGraphicsPipelineState(&pso_desc_scene) }.unwrap();
    }

    // Create a sphere
    #[repr(C)]
    struct vertex_element([f32; 3], [f32; 3]);
    #[repr(C)]
    struct quad_index_list([u16; 6]);
    let mut vertices: Vec<vertex_element> = Vec::new();
    vertices.reserve(((SPHERE_STACKS + 1) * (SPHERE_SLICES + 1)) as usize);
    for y in 0..=SPHERE_STACKS {
        for x in 0..=SPHERE_SLICES {
            let v0 = x as f32 / SPHERE_SLICES as f32;
            let v1 = y as f32 / SPHERE_STACKS as f32;
            let theta = 2.0 * PI * v0;
            let phi = 2.0 * PI * v1 / 2.0;
            let pos = [phi.sin() * theta.sin(), phi.cos(), phi.sin() * theta.cos()];
            let r = 1.0f32;
            let norm = [pos[0] / r, pos[1] / r, pos[2] / r];
            vertices.push(vertex_element(pos, norm));
        }
    }
    let mut indices: Vec<quad_index_list> = Vec::new();
    indices.reserve((SPHERE_STACKS * SPHERE_SLICES) as usize);
    for y in 0..SPHERE_STACKS {
        for x in 0..SPHERE_SLICES {
            let b: u16 = (y * (SPHERE_SLICES + 1) + x).try_into().unwrap();
            let s: u16 = (SPHERE_SLICES + 1).try_into().unwrap();
            //indices.push(quad_index_list([b, b + s, b + 1, b + s, b + s + 1, b + 1]));
            indices.push(quad_index_list([b, b + 1, b + s, b + 1, b + s + 1, b + s]));
        }
    }
    let vb_size = std::mem::size_of::<vertex_element>() * vertices.len();
    let ib_size = std::mem::size_of::<quad_index_list>() * indices.len();

    // Craete resources
    let vb_ib: ID3D12Resource = {
        let mut desc: D3D12_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = (vb_size + ib_size).try_into().unwrap();
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        let mut heap: D3D12_HEAP_PROPERTIES = unsafe { std::mem::zeroed() };
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        let mut res: Option<ID3D12Resource> = None;
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut res) }.unwrap();
        res.unwrap()
    };
    let vb_view = D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: unsafe { vb_ib.GetGPUVirtualAddress() },
        SizeInBytes: vb_size.try_into().unwrap(),
        StrideInBytes: std::mem::size_of::<vertex_element>() as u32,
    };
    let ib_view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: vb_view.BufferLocation + vb_view.SizeInBytes as u64,
        SizeInBytes: ib_size.try_into().unwrap(),
        Format: DXGI_FORMAT_R16_UINT,
    };
    let ib_count: u32 = 6/*indices per quad*/ * indices.len() as u32;
    let cb: [_; BUFFER_COUNT as usize] = {
        array_init::array_init(|_| {
            let mut desc: D3D12_RESOURCE_DESC = unsafe { std::mem::zeroed() };
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = 1 * 1024 * 1024;
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            let mut heap: D3D12_HEAP_PROPERTIES = unsafe { std::mem::zeroed() };
            heap.Type = D3D12_HEAP_TYPE_UPLOAD;
            let mut res: Option<ID3D12Resource> = None;
            unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut res) }.unwrap();
            res.unwrap()
        })
    };

    // Upload data
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe { vb_ib.Map(0, None, Some(&mut p)) }.unwrap();
    unsafe { libc::memcpy(p, vertices.as_ptr() as _, vb_size) };
    unsafe { libc::memcpy((p as usize + vb_size) as *mut c_void, indices.as_ptr() as _, ib_size) };
    unsafe { vb_ib.Unmap(0, None) };

    unsafe { SetDllDirectoryW(PCWSTR(&0u16)) };

    Resource { width, height, rootsig, pso, vb_ib, vb_view, ib_view, ib_count, cb }
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
        let dsv_stride = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) } as usize;
        let res_stride = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as usize;
        let sampler_stride = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) } as usize;

        let cmd_alloc: [_; BUFFER_COUNT as usize] =
            array_init::array_init(|_: usize| -> ID3D12CommandAllocator {
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap()
            });
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
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            device.CreateDescriptorHeap(&desc)
        }.unwrap();
        let swap_chain_tex: [_; BUFFER_COUNT as usize] =
            array_init::array_init(|i| -> ID3D12Resource {
                unsafe { swap_chain.GetBuffer(i as u32) }.unwrap()
            });
        let h_rtv = unsafe { swap_chain_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in swap_chain_tex.iter().enumerate() {
            let desc = D3D12_CPU_DESCRIPTOR_HANDLE{
                ptr: h_rtv.ptr + i.0 * rtv_stride
            };
            unsafe { device.CreateRenderTargetView(i.1, None, desc) };
        }

        let resource = create_resources(&device, width, height);
        D3D{
            dxgi_factory: factory,
            device,
            rtv_stride,
            dsv_stride,
            res_stride,
            sampler_stride,
            cmd_alloc,
            cmd_queue,
            swap_chain,
            cmd_list,
            frame_count: 0,
            fence,
            swap_chain_tex: Some(swap_chain_tex),
            swap_chain_heap,
            is_fullscreen: false,
            scene: unsafe { std::mem::zeroed() },
            resource,
        }
    }
}

impl D3DBase for D3D {
    fn draw(&mut self) {
        self.frame_count += 1;
        let frame_index = unsafe { self.swap_chain.GetCurrentBackBufferIndex() };
        let swap_chain_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { self.swap_chain_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
                + frame_index as usize * self.rtv_stride,
        };

        let cb_scene = &self.resource.cb[self.frame_count as usize % 3];
        let mut p_cb_scene: *mut c_void = std::ptr::null_mut();
        unsafe { cb_scene.Map(0, None, Some(&mut p_cb_scene)) }.unwrap();

        struct CBScene {
            view_proj: XMMATRIX,
        }
        let scene_data = CBScene {
            view_proj: XMMatrixIdentity(),
        };
        unsafe { libc::memcpy(p_cb_scene, &scene_data as *const CBScene as _, std::mem::size_of::<CBScene>()) };

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

        unsafe { self.cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        unsafe { self.cmd_list.IASetVertexBuffers(0, Some(&[self.resource.vb_view])) };
        unsafe { self.cmd_list.IASetIndexBuffer(Some(&self.resource.ib_view)) };
        unsafe { self.cmd_list.SetPipelineState(&self.resource.pso) };
        unsafe { self.cmd_list.SetGraphicsRootSignature(&self.resource.rootsig) };
        unsafe { self.cmd_list.SetGraphicsRootConstantBufferView(0, cb_scene.GetGPUVirtualAddress()) };
        let rect = RECT {
            left: 0, top: 0, right: self.resource.width as i32, bottom: self.resource.height as i32
        };
        unsafe { self.cmd_list.RSSetScissorRects(&[rect]) };
        let viewport = D3D12_VIEWPORT {
            Width: self.resource.width as f32, Height: self.resource.height as f32, MaxDepth: 1.0f32, ..Default::default()
        };
        unsafe { self.cmd_list.RSSetViewports(&[viewport]) };
        unsafe { self.cmd_list.OMSetRenderTargets(1, Some(&swap_chain_rtv), BOOL(0), None) };
        unsafe { self.cmd_list.DrawIndexedInstanced(self.resource.ib_count, 1, 0, 0, 0) };

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
                array_init::array_init(|i| -> ID3D12Resource {
                    unsafe { self.swap_chain.GetBuffer(i as u32) }.unwrap()
                });
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

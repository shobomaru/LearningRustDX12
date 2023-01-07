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
        ((a + b - 1) & !(b - 1)).try_into().unwrap()
    }}
}

pub trait D3DBase {
    fn draw(&mut self);
    fn present(&mut self) -> Result<()>;
    fn wait(&mut self) -> Result<()>;
}

pub trait D3DTest {
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
const DEFAULT_DEPTH_CLEAR: f32 = 0.0; // Reversed depth

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
    #[allow(dead_code)]
    copy_fence: ID3D12Fence,
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
    tex_sailboat: ID3D12Resource,
    #[allow(dead_code)]
    z_tex: ID3D12Resource,
    dsv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    sampler_heap: ID3D12DescriptorHeap,
}

fn create_scene() -> Scene {
    Scene {
        camera_pos: XMVectorSet(0.0, 4.0, -4.0, 0.0),
        camera_target: XMVectorSet(0.0, 0.0, 0.0, 0.0),
        camera_up: XMVectorSet(0.0, 1.0, 0.0, 0.0),
        camera_fov: XMConvertToRadians(45.0),
        camera_near: 0.01,
        camera_far: 100.0,
    }
}

const SPHERE_STACKS: u32 = 10;
const SPHERE_SLICES: u32 = 12;

fn create_resources(device: &ID3D12Device, width: u32, height: u32) -> Resource {
    // Create a root signature
    let rootsig: ID3D12RootSignature = {
        let desc_range_tex = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 10,
            OffsetInDescriptorsFromTableStart: 0,
            ..Default::default()
        };
        let desc_range_sampler = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: 10,
            OffsetInDescriptorsFromTableStart: 0,
            ..Default::default()
        };

        let root_param_tex = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &desc_range_tex,
                }
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let root_param_cbv = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0, RegisterSpace: 0
                }
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        };
        let root_param_sampler = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &desc_range_sampler,
                }
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let root_params = [root_param_tex, root_param_cbv, root_param_sampler];
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: root_params.len() as u32,
            pParameters: root_params.as_ptr(),
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
    float2 texcoord : Texcoord;
};
Output main(float3 position : Position, float3 normal : Normal, float2 texcoord : Texcoord) {
    Output output;
    output.position = mul(float4(position, 1), ViewProj);
    output.world = position;
    output.normal = normalize(normal);
    output.texcoord = texcoord;
    return output;
}
"#;

        let code_scene_ps = r#"
struct Input {
    float4 position : SV_Position;
    float3 world : World;
    float3 normal : Normal;
    float2 texcoord : Texcoord;
};
Texture2D Tex;
SamplerState SS;
float4 main(Input input) : SV_Target {
    float4 color = Tex.Sample(SS, input.texcoord);
    return color;
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
            D3D12_INPUT_ELEMENT_DESC{
                SemanticName: PCSTR("TEXCOORD\0".as_ptr()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 24,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];
        let mut rs_desc: D3D12_RASTERIZER_DESC = unsafe { std::mem::zeroed() };
        rs_desc.CullMode = D3D12_CULL_MODE_BACK;
        rs_desc.FillMode = D3D12_FILL_MODE_SOLID;
        rs_desc.DepthClipEnable = BOOL(1);
        let mut ds_desc: D3D12_DEPTH_STENCIL_DESC = unsafe { std::mem::zeroed() };
        ds_desc.DepthEnable = BOOL(1);
        ds_desc.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        ds_desc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
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
    struct vertex_element([f32; 3], [f32; 3], [f32; 2]);
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
            let uv = [v0, v1];
            vertices.push(vertex_element(pos, norm, uv));
        }
    }
    let mut indices: Vec<quad_index_list> = Vec::new();
    indices.reserve((SPHERE_STACKS * SPHERE_SLICES) as usize);
    for y in 0..SPHERE_STACKS {
        for x in 0..SPHERE_SLICES {
            let b: u16 = (y * (SPHERE_SLICES + 1) + x).try_into().unwrap();
            let s: u16 = (SPHERE_SLICES + 1).try_into().unwrap();
            indices.push(quad_index_list([b, b + 1, b + s, b + 1, b + s + 1, b + s]));
        }
    }
    let vb_size = std::mem::size_of::<vertex_element>() * vertices.len();
    let ib_size = std::mem::size_of::<quad_index_list>() * indices.len();

    // Craete scene resources
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

    // Create scene textures

    let z_tex: ID3D12Resource = {
        let mut desc: D3D12_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Format = DXGI_FORMAT_D32_FLOAT;
        desc.Width = width.try_into().unwrap();
        desc.Height = height.try_into().unwrap();
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        let mut heap: D3D12_HEAP_PROPERTIES = unsafe { std::mem::zeroed() };
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        let mut res: Option<ID3D12Resource> = None;
        let cval = D3D12_CLEAR_VALUE {
            Format: desc.Format,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE{ Depth: DEFAULT_DEPTH_CLEAR, Stencil: 0 }
            }
        };
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&cval), &mut res) }.unwrap();
        res.unwrap()
    };
    let dsv_heap: ID3D12DescriptorHeap = {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 10,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        unsafe { device.CreateDescriptorHeap(&desc) }.unwrap()
    };
    {
        let h = unsafe { dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe { device.CreateDepthStencilView(&z_tex, None, h) };
    }

    // Load images

    struct ImageData {
        extent: [u32; 3],
        data: Vec<u8>,
    }

    #[repr(C, packed(1))]
    #[derive(Default)]
    struct BitmapFileHeader(u16, u32, u16, u16, u32);
    #[repr(C)]
    #[derive(Debug, Clone, Default)]
    struct BitmapInfoHeader {
        size: u32, width: i32, height: i32, planes: u16, bit_count: u16, compression: u32,
        size_image: u32, x_pels_per_meter: i32, y_pels_per_meter: i32, clr_used: u32, clr_important: u32
    }

    let load_bitmap = |src: &[u8]| {
        let mut p = src.as_ptr();
        let magic = unsafe { *(p as *const u16) };
        assert_eq!(magic, 0x4D42); // BMP header
        p = (p as usize + std::mem::size_of::<BitmapFileHeader>()) as *const u8; // 14byte
        // Copy info header into local stack in order to avoid misaligned read
        let mut info_data = [0u8; std::mem::size_of::<BitmapInfoHeader>()];
        unsafe { libc::memcpy(info_data.as_mut_ptr() as _, p as _, info_data.len()) };
        let info: &BitmapInfoHeader = unsafe { &std::slice::from_raw_parts(info_data.as_ptr() as *const BitmapInfoHeader, 1)[0] };
        p = (p as usize + std::mem::size_of::<BitmapInfoHeader>()) as *const u8; // 40byte

        let mut oneline: Vec<u8> = Vec::new();
        oneline.resize(info.width as usize * 3, 0); // rgb
        let mut data: Vec<u8> = Vec::new();
        data.resize(info.width as usize * info.height as usize * 4, 0);
        for y in 0..info.height {
            unsafe { libc::memcpy(oneline.as_mut_ptr() as _, p as _, oneline.len()) };
            p = (p as usize + oneline.len()) as *const u8;
            for x in 0..info.width {
                let q = (info.height - 1 - y) * info.width + x;
                data[q as usize * 4] = oneline[x as usize * 3 + 2];
                data[q as usize * 4 + 1] = oneline[x as usize * 3 + 1];
                data[q as usize * 4 + 2] = oneline[x as usize * 3];
                data[q as usize * 4 + 3] = 0xFF;
            }
        }
        ImageData{ extent: [info.width as _, info.height as _, 1], data }
    };
    let generate_mipmap = |mip0: ImageData| {
        let downsample = |high: &ImageData| {
            let ext: [u32; 3] = [1.max(high.extent[0] / 2), 1.max(high.extent[1] / 2), 1];
            let mut data: Vec<u8> = Vec::new();
            data.resize(ext[0] as usize * ext[1] as usize * 4, 0);
            for y in 0..ext[1] {
                for x in 0..ext[0] { 
                    let pd = y * ext[0] + x;
                    let ps = 2 * y * 2 * ext[0] + 2 * x;
                    for c in 0..4 {
                        let mut d = high.data[ps as usize * 4 + c] as u32;
                        d += high.data[ps as usize * 4 + 1 * 4 * c] as u32;
                        d += high.data[ps as usize * 4 + high.extent[0] as usize * 4 + c] as u32;
                        d += high.data[ps as usize * 4 + high.extent[0] as usize * 4 + 1 * 4 + c] as u32;
                        data[pd as usize * 4 + c] = ((d + 4) / 2).min(u8::MAX as _).try_into().unwrap();
                    }
                }
            }
            ImageData{ extent: ext, data }
        };
        let mut v: Vec<ImageData> = Vec::new();
        v.push(mip0);
        loop {
            let prev = v.last().unwrap();
            if prev.extent[0] == 1 || prev.extent[1] == 1 {
                break;
            }
            v.push(downsample(prev));
        }
        v
    };

    let sailboat_data = generate_mipmap(load_bitmap(include_bytes!("..\\..\\res\\Sailboat.bmp")));

    let tex_sailboat: ID3D12Resource = {
        let mut desc: D3D12_RESOURCE_DESC = unsafe { std::mem::zeroed() };
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.Width = sailboat_data[0].extent[0] as u64;
        desc.Height = sailboat_data[0].extent[1];
        desc.DepthOrArraySize = 1;
        desc.MipLevels = sailboat_data.len() as u16;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut heap: D3D12_HEAP_PROPERTIES = unsafe { std::mem::zeroed() };
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        let mut res: Option<ID3D12Resource> = None;
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res) }.unwrap();
        res.unwrap()
    };
    
    let srv_heap: ID3D12DescriptorHeap = {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 100,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        unsafe { device.CreateDescriptorHeap(&desc) }.unwrap()
    };
    {
        let h = unsafe { srv_heap.GetCPUDescriptorHandleForHeapStart() };
        // slot 0
        unsafe { device.CreateShaderResourceView(&tex_sailboat, None, h) };
    }
    
    let sampler_heap: ID3D12DescriptorHeap = {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: 100,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        unsafe { device.CreateDescriptorHeap(&desc) }.unwrap()
    };
    {
        let h = unsafe { sampler_heap.GetCPUDescriptorHandleForHeapStart() };
        // slot 0
        let desc = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_ANISOTROPIC,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 4,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
        };
        unsafe { device.CreateSampler(&desc, h) };
    }

    // Upload images
    let copy_fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    {
        let upload_buf: ID3D12Resource = {
            let mut desc: D3D12_RESOURCE_DESC = unsafe { std::mem::zeroed() };
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = 4 * 1024 * 1024;
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
        let mut upload_ptr: *mut c_void = std::ptr::null_mut();
        unsafe { upload_buf.Map(0, None, Some(&mut upload_ptr)) }.unwrap();

        let mut upload_ofs: Vec<usize> = Vec::new();
        upload_ofs.resize(sailboat_data.len() + 1, 0);
        for subres in 0..sailboat_data.len() {
            let src_ptr = sailboat_data[subres].data.as_ptr();
            let actual_row = sailboat_data[subres].extent[0] * 4;
            let aligned_row: usize = align!(actual_row, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
            for y in 0..sailboat_data[subres].extent[1] {
                let src_ofs: usize = y as usize * actual_row as usize;
                let upload_ofs: usize = upload_ofs[subres] + y as usize * aligned_row;
                unsafe { libc::memcpy(
                    (upload_ptr as usize + upload_ofs) as _,
                    (src_ptr as usize + src_ofs) as _,
                    actual_row as usize) };
            }
            let actual_slice = aligned_row * sailboat_data[subres].extent[1] as usize;
            let aligned_slice: usize = align!(actual_slice, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            upload_ofs[subres + 1] = upload_ofs[subres] + aligned_slice;
        }

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default()
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }.unwrap();
        let alloc: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY) }.unwrap();
        let list: ID3D12GraphicsCommandList = unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &alloc, None) }.unwrap();
        
        for i in 0..sailboat_data.len() {
            let loc_dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: Some(tex_sailboat.clone()),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32
                }
            };
            let loc_src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: Some(upload_buf.clone()),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: upload_ofs[i] as u64,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: sailboat_data[i].extent[0] as u32,
                            Height: sailboat_data[i].extent[1] as u32,
                            Depth: 1,
                            RowPitch: align!(sailboat_data[i].extent[0] * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT),
                        }
                    }
                }
            };
            unsafe { list.CopyTextureRegion(&loc_dst, 0, 0, 0, &loc_src, None) };
        }

        unsafe { list.Close() }.unwrap();
        let lists = [Some(ID3D12CommandList::from(&list))];
        unsafe { queue.ExecuteCommandLists(&lists) };
        unsafe { queue.Signal(&copy_fence, 1) }.unwrap();
        // Synchronize CPU to release resources safety()
        unsafe { copy_fence.SetEventOnCompletion(1, None) }.unwrap();
    }

    unsafe { SetDllDirectoryW(PCWSTR(&0u16)) };
    Resource { copy_fence, width, height, rootsig, pso, vb_ib, vb_view, ib_view, ib_count, cb, z_tex, dsv_heap, tex_sailboat, srv_heap, sampler_heap }
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
        unsafe { cmd_queue.Wait(&resource.copy_fence, 1) }.unwrap();

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
            scene: create_scene(),
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
        let scene_dsv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { self.resource.dsv_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
        };

        let cb_scene = &self.resource.cb[self.frame_count as usize % 3];
        let mut p_cb_scene: *mut c_void = std::ptr::null_mut();
        unsafe { cb_scene.Map(0, None, Some(&mut p_cb_scene)) }.unwrap();

        #[repr(C)]
        struct CBScene {
            view_proj: XMMATRIX,
        }
        let scene_data: CBScene = {
            let view = XMMatrixLookAtLH(self.scene.camera_pos, self.scene.camera_target, self.scene.camera_up);
            let aspect = self.resource.width as f32 / self.resource.height as f32;
            let proj = XMMatrixPerspectiveFovLH(self.scene.camera_fov, aspect, self.scene.camera_far, self.scene.camera_near);
            let vp = XMMatrixMultiply(view, &proj);
            CBScene {
                view_proj: XMMatrixTranspose(vp),
            }
        };
        unsafe { libc::memcpy(p_cb_scene, &scene_data as *const CBScene as _, std::mem::size_of::<CBScene>()) };

        let cmd_alloc = &self.cmd_alloc[self.frame_count as usize % 3];
        unsafe { cmd_alloc.Reset() }.unwrap();
        unsafe { self.cmd_list.Reset(cmd_alloc, None) }.unwrap();

        unsafe { self.cmd_list.SetDescriptorHeaps(&[Some(self.resource.srv_heap.clone()), Some(self.resource.sampler_heap.clone())]) };
        
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

        let h_dsv = unsafe { self.resource.dsv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe { self.cmd_list.ClearDepthStencilView(h_dsv, D3D12_CLEAR_FLAG_DEPTH, DEFAULT_DEPTH_CLEAR, 0, &[]) };

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
        unsafe { self.cmd_list.SetGraphicsRootDescriptorTable(0, self.resource.srv_heap.GetGPUDescriptorHandleForHeapStart()) };
        unsafe { self.cmd_list.SetGraphicsRootConstantBufferView(1, cb_scene.GetGPUVirtualAddress()) };
        unsafe { self.cmd_list.SetGraphicsRootDescriptorTable(2, self.resource.sampler_heap.GetGPUDescriptorHandleForHeapStart()) };
        let rect = RECT {
            left: 0, top: 0, right: self.resource.width as i32, bottom: self.resource.height as i32
        };
        unsafe { self.cmd_list.RSSetScissorRects(&[rect]) };
        let viewport = D3D12_VIEWPORT {
            Width: self.resource.width as f32, Height: self.resource.height as f32, MaxDepth: 1.0f32, ..Default::default()
        };
        unsafe { self.cmd_list.RSSetViewports(&[viewport]) };
        unsafe { self.cmd_list.OMSetRenderTargets(1, Some(&swap_chain_rtv), BOOL(0), Some(&scene_dsv)) };
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
}

impl D3DTest for D3D
{
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

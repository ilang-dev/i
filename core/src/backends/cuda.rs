use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use compiler::ir::component::Component;
use compiler::ir::parallel_module::ParallelModule;

use crate::loader::Library;

use super::{dylib_ext, load_program, unique_string, Device, Program};

pub struct CudaRuntime {
    _library: Library,
    path: PathBuf,
    alloc: unsafe extern "C" fn(usize) -> *mut f32,
    free: unsafe extern "C" fn(*mut f32),
    copy_from_host: unsafe extern "C" fn(*mut f32, *const f32, usize),
    copy_to_host: unsafe extern "C" fn(*mut f32, *const f32, usize),
}

unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

static RUNTIME: OnceLock<CudaRuntime> = OnceLock::new();

pub fn compile(component: &Component) -> Result<Program, String> {
    let source = render(component)?;
    let dylib_path = build(&source)?;
    unsafe { load_program(dylib_path, Device::I_DEVICE_CUDA) }
}

pub fn render(component: &Component) -> Result<String, String> {
    let module = lower_to_parallel_module(component)?;
    compiler::backends::cuda::render(&module).map_err(|err| err.to_string())
}

pub fn runtime() -> Result<&'static CudaRuntime, String> {
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = build_runtime()?;
    let _ = RUNTIME.set(runtime);
    RUNTIME
        .get()
        .ok_or_else(|| "failed to initialize cuda runtime".to_string())
}

impl CudaRuntime {
    pub unsafe fn alloc(&self, len: usize) -> *mut f32 {
        (self.alloc)(len)
    }

    pub unsafe fn free(&self, data: *mut f32) {
        (self.free)(data);
    }

    pub unsafe fn copy_from_host(&self, dst: *mut f32, src: *const f32, len: usize) {
        (self.copy_from_host)(dst, src, len);
    }

    pub unsafe fn copy_to_host(&self, dst: *mut f32, src: *const f32, len: usize) {
        (self.copy_to_host)(dst, src, len);
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn lower_to_parallel_module(component: &Component) -> Result<ParallelModule, String> {
    let graph =
        compiler::lower::lower_component_to_graph(component).map_err(|err| format!("{err:?}"))?;
    let stages = compiler::lower::lower_node_graph_to_stage_program(&graph)
        .map_err(|err| format!("{err:?}"))?;
    let kernels = compiler::lower::lower_stage_program_to_kernel_program(&stages)
        .map_err(|err| format!("{err:?}"))?;
    let plan = compiler::lower::lower_kernel_program_to_exec_plan(&kernels)
        .map_err(|err| format!("{err:?}"))?;
    compiler::lower::lower_exec_plan_to_parallel_module(&plan).map_err(|err| format!("{err:?}"))
}

fn build_runtime() -> Result<CudaRuntime, String> {
    let path = build(include_str!("cuda/runtime.cu"))?;
    unsafe {
        let library = Library::open(&path)?;
        let alloc =
            library.symbol::<unsafe extern "C" fn(usize) -> *mut f32>(c"i_cuda_tensor_alloc")?;
        let free = library.symbol::<unsafe extern "C" fn(*mut f32)>(c"i_cuda_tensor_free")?;
        let copy_from_host = library.symbol::<unsafe extern "C" fn(*mut f32, *const f32, usize)>(
            c"i_cuda_tensor_copy_from_host",
        )?;
        let copy_to_host = library.symbol::<unsafe extern "C" fn(*mut f32, *const f32, usize)>(
            c"i_cuda_tensor_copy_to_host",
        )?;
        Ok(CudaRuntime {
            _library: library,
            path,
            alloc,
            free,
            copy_from_host,
            copy_to_host,
        })
    }
}

fn build(source: &str) -> Result<PathBuf, String> {
    let stem = format!("ilang_cuda_{}", unique_string());
    let source_path = std::env::temp_dir().join(format!("{stem}.cu"));
    let dylib_path = std::env::temp_dir().join(format!("{stem}.{}", dylib_ext()));

    std::fs::write(&source_path, source).map_err(|err| err.to_string())?;
    let nvcc = std::env::var("ILANG_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let exit = Command::new(nvcc)
        .args([
            "-O3",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "--diag-suppress=177",
            "--cudart=shared",
        ])
        .arg(&source_path)
        .arg("-o")
        .arg(&dylib_path)
        .status()
        .map_err(|err| err.to_string())?;

    let _ = std::fs::remove_file(&source_path);
    if !exit.success() {
        return Err(format!("nvcc failed with status {exit}"));
    }

    Ok(dylib_path)
}

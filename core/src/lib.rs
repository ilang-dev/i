mod loader;

use loader::Library;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::process::Command;
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

use c2::ir::component::Component;
use c2::ir::module::Module;

#[repr(C)]
pub struct i_tensor {
    pub data: *const f32,
    pub shape: *const usize,
    pub rank: usize,
}

#[repr(C)]
pub struct i_tensor_mut {
    pub data: *mut f32,
    pub shape: *const usize,
    pub rank: usize,
}

#[repr(C)]
pub struct i_owned_tensor {
    pub data: *mut f32,
    pub shape: *mut usize,
    pub rank: usize,
    pub len: usize,
}

#[repr(C)]
pub struct i_outputs {
    pub tensors: *mut i_owned_tensor,
    pub count: usize,
}

#[allow(non_camel_case_types)]
pub struct i_component {
    inner: Component,
}

#[allow(non_camel_case_types)]
pub struct i_program {
    _library: Library,
    path: PathBuf,
    count: unsafe extern "C" fn() -> usize,
    ranks: unsafe extern "C" fn(*mut usize),
    shapes: unsafe extern "C" fn(*const i_tensor, *mut *mut usize),
    exec: unsafe extern "C" fn(*const i_tensor, *mut i_tensor_mut),
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn i_parse(src: *const c_char) -> *mut i_component {
    let Some(src) = read_str(src) else {
        return null_with_error("null source");
    };

    match c2::front::parse_component(src) {
        Ok(inner) => Box::into_raw(Box::new(i_component { inner })),
        Err(err) => null_with_error(format!("{err:?}")),
    }
}

#[no_mangle]
pub extern "C" fn i_identity() -> *mut i_component {
    Box::into_raw(Box::new(i_component {
        inner: c2::component::identity(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn i_chain(
    left: *const i_component,
    right: *const i_component,
) -> *mut i_component {
    combine(left, right, Component::chain)
}

#[no_mangle]
pub unsafe extern "C" fn i_compose(
    left: *const i_component,
    right: *const i_component,
) -> *mut i_component {
    combine(left, right, Component::compose)
}

#[no_mangle]
pub unsafe extern "C" fn i_fanout(
    left: *const i_component,
    right: *const i_component,
) -> *mut i_component {
    combine(left, right, Component::fanout)
}

#[no_mangle]
pub unsafe extern "C" fn i_pair(
    left: *const i_component,
    right: *const i_component,
) -> *mut i_component {
    combine(left, right, Component::pair)
}

#[no_mangle]
pub unsafe extern "C" fn i_swap(component: *const i_component) -> *mut i_component {
    let Some(component) = component.as_ref() else {
        return null_with_error("null component");
    };

    Box::into_raw(Box::new(i_component {
        inner: component.inner.clone().swap(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn i_code(component: *const i_component) -> *mut c_char {
    let Some(component) = component.as_ref() else {
        set_error("null component");
        return ptr::null_mut();
    };

    match render_component(&component.inner).and_then(|source| {
        CString::new(source).map_err(|_| "source contains interior NUL".to_string())
    }) {
        Ok(source) => source.into_raw(),
        Err(err) => {
            set_error(err);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_compile(component: *const i_component) -> *mut i_program {
    let Some(component) = component.as_ref() else {
        return null_with_error("null component");
    };

    match compile(&component.inner) {
        Ok(program) => Box::into_raw(Box::new(program)),
        Err(err) => null_with_error(err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_output_count(program: *const i_program) -> usize {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return 0;
    };
    (program.count)()
}

#[no_mangle]
pub unsafe extern "C" fn i_output_ranks(program: *const i_program, ranks: *mut usize) -> i32 {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return -1;
    };
    if ranks.is_null() {
        set_error("null ranks");
        return -1;
    }
    (program.ranks)(ranks);
    0
}

#[no_mangle]
pub unsafe extern "C" fn i_output_shapes(
    program: *const i_program,
    inputs: *const i_tensor,
    input_count: usize,
    shapes: *mut *mut usize,
) -> i32 {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return -1;
    };
    if input_count > 0 && inputs.is_null() {
        set_error("null inputs");
        return -1;
    }
    if shapes.is_null() {
        set_error("null shapes");
        return -1;
    }
    (program.shapes)(inputs, shapes);
    0
}

#[no_mangle]
pub unsafe extern "C" fn i_exec_into(
    program: *const i_program,
    inputs: *const i_tensor,
    input_count: usize,
    outputs: *mut i_tensor_mut,
    output_count: usize,
) -> i32 {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return -1;
    };
    if input_count > 0 && inputs.is_null() {
        set_error("null inputs");
        return -1;
    }
    if output_count != (program.count)() {
        set_error("wrong output count");
        return -1;
    }
    if output_count > 0 && outputs.is_null() {
        set_error("null outputs");
        return -1;
    }
    (program.exec)(inputs, outputs);
    0
}

#[no_mangle]
pub unsafe extern "C" fn i_exec(
    program: *const i_program,
    inputs: *const i_tensor,
    input_count: usize,
) -> i_outputs {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return empty_outputs();
    };
    if input_count > 0 && inputs.is_null() {
        set_error("null inputs");
        return empty_outputs();
    }

    let count = (program.count)();
    let mut ranks = vec![0usize; count];
    (program.ranks)(ranks.as_mut_ptr());

    let mut shapes: Vec<Vec<usize>> = ranks.iter().map(|rank| vec![0; *rank]).collect();
    let mut shape_ptrs: Vec<*mut usize> = shapes.iter_mut().map(Vec::as_mut_ptr).collect();
    (program.shapes)(inputs, shape_ptrs.as_mut_ptr());

    let mut data: Vec<Vec<f32>> = shapes
        .iter()
        .map(|shape| vec![0.0; shape.iter().product()])
        .collect();

    let mut output_views: Vec<i_tensor_mut> = data
        .iter_mut()
        .zip(shapes.iter())
        .map(|(data, shape)| i_tensor_mut {
            data: data.as_mut_ptr(),
            shape: shape.as_ptr(),
            rank: shape.len(),
        })
        .collect();

    (program.exec)(inputs, output_views.as_mut_ptr());

    let mut outputs = Vec::with_capacity(count);
    for (mut data, mut shape) in data.into_iter().zip(shapes) {
        let tensor = i_owned_tensor {
            data: data.as_mut_ptr(),
            shape: shape.as_mut_ptr(),
            rank: shape.len(),
            len: data.len(),
        };
        std::mem::forget(data);
        std::mem::forget(shape);
        outputs.push(tensor);
    }

    let result = i_outputs {
        tensors: outputs.as_mut_ptr(),
        count: outputs.len(),
    };
    std::mem::forget(outputs);
    result
}

#[no_mangle]
pub unsafe extern "C" fn i_component_free(component: *mut i_component) {
    if !component.is_null() {
        drop(Box::from_raw(component));
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_program_free(program: *mut i_program) {
    if !program.is_null() {
        drop(Box::from_raw(program));
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_outputs_free(outputs: i_outputs) {
    if outputs.tensors.is_null() {
        return;
    }

    let tensors = Vec::from_raw_parts(outputs.tensors, outputs.count, outputs.count);
    for tensor in tensors {
        if !tensor.data.is_null() {
            drop(Vec::from_raw_parts(tensor.data, tensor.len, tensor.len));
        }
        if !tensor.shape.is_null() {
            drop(Vec::from_raw_parts(tensor.shape, tensor.rank, tensor.rank));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

#[no_mangle]
pub extern "C" fn i_error() -> *const c_char {
    LAST_ERROR.with(|err| {
        err.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

unsafe fn combine(
    left: *const i_component,
    right: *const i_component,
    f: impl FnOnce(Component, Component) -> Component,
) -> *mut i_component {
    let Some(left) = left.as_ref() else {
        return null_with_error("null left component");
    };
    let Some(right) = right.as_ref() else {
        return null_with_error("null right component");
    };

    Box::into_raw(Box::new(i_component {
        inner: f(left.inner.clone(), right.inner.clone()),
    }))
}

fn compile(component: &Component) -> Result<i_program, String> {
    let source = render_component(component)?;
    let dylib_path = build(&source)?;

    unsafe {
        let library = Library::open(&dylib_path)?;
        let count = library.symbol::<unsafe extern "C" fn() -> usize>(c"count")?;
        let ranks = library.symbol::<unsafe extern "C" fn(*mut usize)>(c"ranks")?;
        let shapes =
            library.symbol::<unsafe extern "C" fn(*const i_tensor, *mut *mut usize)>(c"shapes")?;
        let exec =
            library.symbol::<unsafe extern "C" fn(*const i_tensor, *mut i_tensor_mut)>(c"exec")?;

        Ok(i_program {
            _library: library,
            path: dylib_path,
            count,
            ranks,
            shapes,
            exec,
        })
    }
}

fn render_component(component: &Component) -> Result<String, String> {
    let module = lower_component_to_module(component)?;
    Ok(c2::backends::c::render(&module))
}

fn lower_component_to_module(component: &Component) -> Result<Module, String> {
    let graph = c2::lower::lower_component_to_graph(component).map_err(|err| format!("{err:?}"))?;
    let stages =
        c2::lower::lower_node_graph_to_stage_program(&graph).map_err(|err| format!("{err:?}"))?;
    let kernels = c2::lower::lower_stage_program_to_kernel_program(&stages)
        .map_err(|err| format!("{err:?}"))?;
    let plan =
        c2::lower::lower_kernel_program_to_exec_plan(&kernels).map_err(|err| format!("{err:?}"))?;
    c2::lower::lower_exec_plan_to_module(&plan).map_err(|err| format!("{err:?}"))
}

fn build(source: &str) -> Result<PathBuf, String> {
    let stem = format!("ilang_{}", unique_string());
    let source_path = std::env::temp_dir().join(format!("{stem}.c"));
    let dylib_path = std::env::temp_dir().join(format!("{stem}.{}", dylib_ext()));

    std::fs::write(&source_path, source).map_err(|err| err.to_string())?;
    let exit = Command::new("cc")
        .args(["-O3", "-shared", "-fPIC"])
        .arg(&source_path)
        .arg("-o")
        .arg(&dylib_path)
        .arg("-lm")
        .status()
        .map_err(|err| err.to_string())?;

    let _ = std::fs::remove_file(&source_path);
    if !exit.success() {
        return Err(format!("cc failed with status {exit}"));
    }

    Ok(dylib_path)
}

fn dylib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

fn unique_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string()
}

fn read_str<'a>(s: *const c_char) -> Option<&'a str> {
    if s.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(s).to_str().ok() }
}

fn set_error(message: impl Into<String>) {
    let mut message = message.into();
    message.retain(|c| c != '\0');
    LAST_ERROR.with(|err| *err.borrow_mut() = CString::new(message).ok());
}

fn null_with_error<T>(message: impl Into<String>) -> *mut T {
    set_error(message);
    ptr::null_mut()
}

fn empty_outputs() -> i_outputs {
    i_outputs {
        tensors: ptr::null_mut(),
        count: 0,
    }
}

impl Drop for i_program {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

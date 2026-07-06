mod backends;
mod loader;

use backends::{Device as i_device, Program as i_program};
use compiler::ir::common::BindId;
use compiler::ir::component::Component;
use compiler::ir::graph::Input;
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

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

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum i_input_kind {
    I_INPUT_FREE = 0,
    I_INPUT_BOUND = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct i_input {
    pub kind: i_input_kind,
    /// Meaningful only when `kind == I_INPUT_BOUND`.
    pub id: usize,
}

#[allow(non_camel_case_types)]
pub struct i_component {
    inner: Component,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn i_parse(expr: *const c_char) -> *mut i_component {
    let Some(expr) = read_str(expr) else {
        return null_with_error("null expression");
    };

    match compiler::front::parse_component(expr) {
        Ok(inner) => Box::into_raw(Box::new(i_component { inner })),
        Err(err) => null_with_error(format!("{err:?}")),
    }
}

#[no_mangle]
pub extern "C" fn i_identity() -> *mut i_component {
    Box::into_raw(Box::new(i_component {
        inner: compiler::component::identity(),
    }))
}

#[no_mangle]
pub extern "C" fn i_bind(id: usize) -> *mut i_component {
    Box::into_raw(Box::new(i_component {
        inner: compiler::component::bind(BindId(id)),
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
pub unsafe extern "C" fn i_component_input_count(
    component: *const i_component,
    out: *mut usize,
) -> i32 {
    let Some(component) = component.as_ref() else {
        set_error("null component");
        return -1;
    };
    if out.is_null() {
        set_error("null input count output");
        return -1;
    }

    match component_graph_boundary(&component.inner) {
        Ok((inputs, _outputs)) => {
            *out = inputs.len();
            0
        }
        Err(err) => {
            set_error(err);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_component_output_count(
    component: *const i_component,
    out: *mut usize,
) -> i32 {
    let Some(component) = component.as_ref() else {
        set_error("null component");
        return -1;
    };
    if out.is_null() {
        set_error("null output count output");
        return -1;
    }

    match component_graph_boundary(&component.inner) {
        Ok((_inputs, outputs)) => {
            *out = outputs;
            0
        }
        Err(err) => {
            set_error(err);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_component_inputs(
    component: *const i_component,
    out: *mut i_input,
) -> i32 {
    let Some(component) = component.as_ref() else {
        set_error("null component");
        return -1;
    };

    match component_graph_boundary(&component.inner) {
        Ok((inputs, _outputs)) => {
            if !inputs.is_empty() && out.is_null() {
                set_error("null component inputs output");
                return -1;
            }
            for (index, input) in inputs.iter().enumerate() {
                *out.add(index) = match input {
                    Input::Free => i_input {
                        kind: i_input_kind::I_INPUT_FREE,
                        id: 0,
                    },
                    Input::Bound(id) => i_input {
                        kind: i_input_kind::I_INPUT_BOUND,
                        id: id.0,
                    },
                };
            }
            0
        }
        Err(err) => {
            set_error(err);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_code(component: *const i_component, device: i_device) -> *mut c_char {
    let Some(component) = component.as_ref() else {
        set_error("null component");
        return ptr::null_mut();
    };

    match render_component(&component.inner, device).and_then(|source| {
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
pub unsafe extern "C" fn i_compile(
    component: *const i_component,
    device: i_device,
) -> *mut i_program {
    let Some(component) = component.as_ref() else {
        return null_with_error("null component");
    };

    match compile(&component.inner, device) {
        Ok(program) => Box::into_raw(Box::new(program)),
        Err(err) => null_with_error(err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_program_device(program: *const i_program) -> i_device {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return i_device::I_DEVICE_CPU;
    };
    program.device()
}

#[no_mangle]
pub unsafe extern "C" fn i_alloc(device: i_device, len: usize) -> *mut f32 {
    match device {
        i_device::I_DEVICE_CPU => {
            let Some(bytes) = len.checked_mul(std::mem::size_of::<f32>()) else {
                return null_with_error("allocation size overflow");
            };
            let ptr = malloc(bytes.max(1)) as *mut f32;
            if ptr.is_null() {
                set_error("allocation failed");
            }
            ptr
        }
        i_device::I_DEVICE_CUDA => match backends::cuda::runtime() {
            Ok(runtime) => {
                let ptr = runtime.alloc(len);
                if ptr.is_null() {
                    set_error("allocation failed");
                }
                ptr
            }
            Err(err) => null_with_error(err),
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_free(device: i_device, data: *mut f32) {
    if data.is_null() {
        return;
    }

    match device {
        i_device::I_DEVICE_CPU => free(data.cast::<c_void>()),
        i_device::I_DEVICE_CUDA => match backends::cuda::runtime() {
            Ok(runtime) => runtime.free(data),
            Err(err) => set_error(err),
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_copy(
    dst_device: i_device,
    dst: *mut f32,
    src_device: i_device,
    src: *const f32,
    len: usize,
) -> i32 {
    if len > 0 && (dst.is_null() || src.is_null()) {
        set_error("null copy pointer");
        return -1;
    }
    if len == 0 {
        return 0;
    }

    match (dst_device, src_device) {
        (i_device::I_DEVICE_CPU, i_device::I_DEVICE_CPU) => {
            ptr::copy_nonoverlapping(src, dst, len);
            0
        }
        (i_device::I_DEVICE_CUDA, i_device::I_DEVICE_CPU) => match backends::cuda::runtime() {
            Ok(runtime) => {
                runtime.copy_from_host(dst, src, len);
                0
            }
            Err(err) => {
                set_error(err);
                -1
            }
        },
        (i_device::I_DEVICE_CPU, i_device::I_DEVICE_CUDA) => match backends::cuda::runtime() {
            Ok(runtime) => {
                runtime.copy_to_host(dst, src, len);
                0
            }
            Err(err) => {
                set_error(err);
                -1
            }
        },
        (i_device::I_DEVICE_CUDA, i_device::I_DEVICE_CUDA) => match backends::cuda::runtime() {
            Ok(runtime) => {
                runtime.copy(dst, src, len);
                0
            }
            Err(err) => {
                set_error(err);
                -1
            }
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn i_output_count(program: *const i_program) -> usize {
    let Some(program) = program.as_ref() else {
        set_error("null program");
        return 0;
    };
    program.count()
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
    program.ranks(ranks);
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
    program.shapes(inputs, shapes);
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
    if output_count != program.count() {
        set_error("wrong output count");
        return -1;
    }
    if output_count > 0 && outputs.is_null() {
        set_error("null outputs");
        return -1;
    }
    program.exec(inputs, outputs);
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

    let count = program.count();
    let mut ranks = vec![0usize; count];
    program.ranks(ranks.as_mut_ptr());

    let mut shapes: Vec<Vec<usize>> = ranks.iter().map(|rank| vec![0; *rank]).collect();
    let mut shape_ptrs: Vec<*mut usize> = shapes.iter_mut().map(Vec::as_mut_ptr).collect();
    program.shapes(inputs, shape_ptrs.as_mut_ptr());

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

    program.exec(inputs, output_views.as_mut_ptr());

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

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
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

fn compile(component: &Component, device: i_device) -> Result<i_program, String> {
    match device {
        i_device::I_DEVICE_CPU => backends::cpu::compile(component),
        i_device::I_DEVICE_CUDA => backends::cuda::compile(component),
    }
}

fn render_component(component: &Component, device: i_device) -> Result<String, String> {
    match device {
        i_device::I_DEVICE_CPU => backends::cpu::render(component),
        i_device::I_DEVICE_CUDA => backends::cuda::render(component),
    }
}

fn component_graph_boundary(component: &Component) -> Result<(Vec<Input>, usize), String> {
    let graph =
        compiler::lower::lower_component_to_graph(component).map_err(|err| err.to_string())?;
    Ok((graph.inputs, graph.outputs.len()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_deduplicated_bound_inputs() {
        unsafe {
            let left = i_bind(7);
            let right = i_bind(7);
            let pair = i_pair(left, right);
            assert!(!pair.is_null());

            let mut count = 0usize;
            assert_eq!(i_component_input_count(pair, &mut count), 0);
            assert_eq!(count, 1);

            let mut inputs = [i_input {
                kind: i_input_kind::I_INPUT_FREE,
                id: 0,
            }];
            assert_eq!(i_component_inputs(pair, inputs.as_mut_ptr()), 0);
            assert_eq!(inputs[0].kind, i_input_kind::I_INPUT_BOUND);
            assert_eq!(inputs[0].id, 7);

            i_component_free(left);
            i_component_free(right);
            i_component_free(pair);
        }
    }
}

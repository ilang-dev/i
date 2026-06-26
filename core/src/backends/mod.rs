pub mod cpu;
pub mod cuda;

use std::path::PathBuf;

use crate::loader::Library;
use crate::{i_tensor, i_tensor_mut};

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    I_DEVICE_CPU = 0,
    I_DEVICE_CUDA = 1,
}

#[allow(non_camel_case_types)]
pub struct Program {
    pub(crate) _library: Library,
    pub(crate) path: PathBuf,
    pub(crate) device: Device,
    pub(crate) count: unsafe extern "C" fn() -> usize,
    pub(crate) ranks: unsafe extern "C" fn(*mut usize),
    pub(crate) shapes: unsafe extern "C" fn(*const i_tensor, *mut *mut usize),
    pub(crate) exec: unsafe extern "C" fn(*const i_tensor, *mut i_tensor_mut),
}

impl Program {
    pub fn device(&self) -> Device {
        self.device
    }

    pub unsafe fn count(&self) -> usize {
        (self.count)()
    }

    pub unsafe fn ranks(&self, ranks: *mut usize) {
        (self.ranks)(ranks);
    }

    pub unsafe fn shapes(&self, inputs: *const i_tensor, shapes: *mut *mut usize) {
        (self.shapes)(inputs, shapes);
    }

    pub unsafe fn exec(&self, inputs: *const i_tensor, outputs: *mut i_tensor_mut) {
        (self.exec)(inputs, outputs);
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(crate) unsafe fn load_program(dylib_path: PathBuf, device: Device) -> Result<Program, String> {
    let library = Library::open(&dylib_path)?;
    let count = library.symbol::<unsafe extern "C" fn() -> usize>(c"count")?;
    let ranks = library.symbol::<unsafe extern "C" fn(*mut usize)>(c"ranks")?;
    let shapes =
        library.symbol::<unsafe extern "C" fn(*const i_tensor, *mut *mut usize)>(c"shapes")?;
    let exec =
        library.symbol::<unsafe extern "C" fn(*const i_tensor, *mut i_tensor_mut)>(c"exec")?;

    Ok(Program {
        _library: library,
        path: dylib_path,
        device,
        count,
        ranks,
        shapes,
        exec,
    })
}

pub(crate) fn dylib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

pub(crate) fn unique_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string()
}

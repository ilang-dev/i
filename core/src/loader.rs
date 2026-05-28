use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

pub struct Library {
    handle: *mut c_void,
}

impl Library {
    pub unsafe fn open(path: &Path) -> Result<Self, String> {
        let path = path
            .to_str()
            .ok_or_else(|| "library path is not valid UTF-8".to_string())?;
        let path = CString::new(path).map_err(|_| "library path contains NUL".to_string())?;
        let handle = dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL);
        if handle.is_null() {
            return Err(dlerror_string());
        }
        Ok(Self { handle })
    }

    pub unsafe fn symbol<T: Copy>(&self, name: &CStr) -> Result<T, String> {
        clear_dlerror();
        let symbol = dlsym(self.handle, name.as_ptr());
        if symbol.is_null() {
            let err = dlerror_string();
            if !err.is_empty() {
                return Err(err);
            }
        }
        Ok(std::mem::transmute_copy(&symbol))
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                dlclose(self.handle);
            }
        }
    }
}

unsafe fn clear_dlerror() {
    let _ = dlerror();
}

unsafe fn dlerror_string() -> String {
    let err = dlerror();
    if err.is_null() {
        return String::new();
    }
    CStr::from_ptr(err).to_string_lossy().into_owned()
}

const RTLD_NOW: i32 = 0x2;
const RTLD_LOCAL: i32 = 0x0;

extern "C" {
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

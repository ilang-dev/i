use std::{io::Error, path::PathBuf};

use crate::block::Program;

pub mod c;
//pub mod cuda;

pub use c::CBackend;
//pub use cuda::CudaBackend;

pub trait Render {
    fn render(program: &Program) -> String;
}

#[allow(dead_code)]
// Not dead code, but the compiler thinks so...?
pub trait Build {
    fn build(source: &str) -> Result<PathBuf, Error>;
}

#[allow(dead_code)]
// Not dead code, but the compiler thinks so...?
pub trait Backend: Render + Build {}

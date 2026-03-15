use super::common::{BufferId, KernelId, Shape};
use super::kernel_loop::Kernel;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<Kernel>,
    pub entry: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    pub shape: Shape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc(BufferId),
    Call {
        kernel: KernelId,
        args: Vec<BufferId>,
    },
    Free(BufferId),
}

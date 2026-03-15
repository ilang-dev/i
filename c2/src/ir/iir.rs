use super::common::{BufferId, KernelId};
use super::kernel_loop::{Alloc, Buffer, Kernel};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<Kernel>,
    pub entry: Entry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub inputs: Vec<BufferId>,
    pub outputs: Vec<BufferId>,
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc(Alloc),
    Call(Call),
    Free(BufferId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Call {
    pub kernel: KernelId,
    pub args: Vec<BufferId>,
}

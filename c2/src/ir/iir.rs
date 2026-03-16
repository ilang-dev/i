use super::common::{BufferId, KernelId, TensorType};
use super::loop_ir::Kernel;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<TensorType>,
    pub kernels: Vec<Kernel>,
    pub entry: Entry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc {
        buffer: BufferId,
        ty: TensorType,
    },
    Call {
        kernel: KernelId,
        reads: Vec<BufferId>,
        writes: Vec<BufferId>,
    },
    Free(BufferId),
}

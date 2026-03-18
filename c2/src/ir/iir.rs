use super::common::{BufferId, ExtentSource, KernelId, Scalar, TensorType};
use super::loop_ir::Kernel;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub shapes: ShapeData,
    pub kernels: Vec<Kernel>,
    pub exec: Exec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeData {
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<OutputTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputTensor {
    pub scalar: Scalar,
    pub shape: Vec<ExtentSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Exec {
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

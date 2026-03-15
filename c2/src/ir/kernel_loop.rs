use super::common::{BufferId, KernelId, LinearExpr, LoopId, Op, StageId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    pub id: KernelId,
    pub params: Vec<BufferId>,
    pub buffers: Vec<Buffer>,
    pub body: Block,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    pub kind: BufferKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferKind {
    Input,
    Output,
    Temporary,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Block {
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc {
        buffer: BufferId,
        shape: Vec<LinearExpr>,
    },
    Loop(Loop),
    Stage(Stage),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Loop {
    pub id: LoopId,
    pub extent: LinearExpr,
    pub body: Block,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub stage: StageId,
    pub kind: StageKind,
    pub op: Op,
    pub inputs: Vec<Use>,
    pub output: Use,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageKind {
    Compute,
    Init,
    Update,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub buffer: BufferId,
    pub index: Vec<LinearExpr>,
}

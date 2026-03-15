use super::common::{BufferId, IndexExpr, KernelId, LoopId, StageId, TensorType};
use super::component::Operator;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    pub id: KernelId,
    pub params: Vec<BufferId>,
    pub buffers: Vec<Buffer>,
    pub body: Region,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    pub id: BufferId,
    pub ty: TensorType,
    pub kind: BufferKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferKind {
    Input,
    Output,
    Temporary,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Region {
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc(Alloc),
    Loop(Loop),
    Execute(StageAction),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Alloc {
    pub buffer: BufferId,
    pub shape: Vec<IndexExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Loop {
    pub id: LoopId,
    pub extent: IndexExpr,
    pub body: Region,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageAction {
    pub stage: StageId,
    pub kind: ActionKind,
    pub op: Operator,
    pub inputs: Vec<BufferUse>,
    pub output: BufferUse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionKind {
    Compute,
    Init,
    Update,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferUse {
    pub buffer: BufferId,
    pub access: Vec<IndexExpr>,
}

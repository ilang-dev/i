use super::common::{LoopId, Op, StageId};

pub type ReadSlot = usize;
pub type WriteSlot = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    pub body: Block,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Block {
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Loop(Loop),
    Action(Action),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Loop {
    pub id: LoopId,
    pub extent: LinearExpr,
    pub body: Block,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Action {
    pub stage: StageId,
    pub kind: ActionKind,
    pub op: Op,
    pub inputs: Vec<Access>,
    pub output: Access,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionKind {
    Compute,
    Init,
    Update,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Access {
    pub buffer: Buffer,
    pub index: Vec<LinearExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Buffer {
    Read(ReadSlot),
    Write(WriteSlot),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LinearExpr {
    pub offset: usize,
    pub terms: Vec<Term>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Term {
    pub loop_id: LoopId,
    pub scale: usize,
}

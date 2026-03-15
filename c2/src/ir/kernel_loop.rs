use super::common::{BufferId, LoopId, Op, StageId, TensorType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    pub params: Vec<Param>,
    pub body: Block,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Param {
    pub kind: ParamKind,
    pub ty: TensorType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParamKind {
    Input,
    Output,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Block {
    pub steps: Vec<Step>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    Alloc(Alloc),
    Loop(Loop),
    Stage(Action),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Alloc {
    pub buffer: BufferId,
    pub ty: TensorType,
    pub shape: Vec<LinearExpr>,
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
    pub inputs: Vec<Use>,
    pub output: Use,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionKind {
    Compute,
    Init,
    Update,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub buffer: BufferId,
    pub index: Vec<LinearExpr>,
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

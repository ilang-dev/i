pub type Axis = char;
pub type ValueId = usize;
pub type StageId = usize;
pub type KernelId = usize;
pub type BufferId = usize;
pub type LoopId = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Extent {
    Known(usize),
    Param(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape(pub Vec<Extent>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern(pub Vec<Axis>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    NoOp,
    Exp,
    Log,
    Sqrt,
    Abs,
    Relu,
    Neg,
    Recip,
    Mul,
    Add,
    Max,
    Min,
    Div,
    Sub,
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

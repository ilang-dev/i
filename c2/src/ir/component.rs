use super::common::{Extent, Symbol};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub outputs: Vec<Component>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Component {
    Expr(Expr),
    Compose {
        left: Box<Component>,
        right: Box<Component>,
    },
    Chain {
        left: Box<Component>,
        right: Box<Component>,
    },
    Fanout {
        left: Box<Component>,
        right: Box<Component>,
    },
    Pair {
        left: Box<Component>,
        right: Box<Component>,
    },
    Swap(Box<Component>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expr {
    pub op: Operator,
    pub inputs: Vec<Pattern>,
    pub output: Pattern,
    pub schedule: ScheduleSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern {
    pub axes: Vec<Symbol>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operator {
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
pub struct ScheduleSpec {
    pub splits: Vec<SplitSpec>,
    pub order: Vec<LoopVar>,
    pub compute_at: Vec<ComputeAtSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitSpec {
    pub axis: Symbol,
    pub factors: Vec<Extent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopVar {
    pub axis: Symbol,
    pub part: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeAtSpec {
    pub input: usize,
    pub at: Option<LoopVar>,
}

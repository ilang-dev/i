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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scalar {
    Float,
    Int,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    pub scalar: Scalar,
    pub shape: Shape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern(pub Vec<Axis>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Split {
    pub axis: Axis,
    pub factors: Vec<Extent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopVar {
    pub axis: Axis,
    pub part: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    NoOp,
    Add,
    Mul,
    Div,
    Sub,
    Max,
    Min,
    Pow,
    Log,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    And,
    Or,
    Xor,
    Not,
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

pub type Axis = char;
pub type ExprId = usize;
pub type ValueId = usize;
pub type StageId = usize;
pub type KernelId = usize;
pub type BufferId = usize;
pub type LoopId = usize;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Index(pub usize);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExtentKind {
    Semantic,
    Base(Vec<usize>),
    Split { level: usize, factor: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Extent {
    Known(usize),
    Param(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExtentSource {
    Const(usize),
    InputDim { input: usize, dim: usize },
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
pub struct AxisRef {
    pub axis: Axis,
    pub part: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
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

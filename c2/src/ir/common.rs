pub type Axis = char;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Index(pub usize);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExtentKind {
    Semantic,
    Base(Vec<usize>),
    Split { level: usize, factor: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DimRef<B> {
    pub buffer: B,
    pub dim: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Extent<B> {
    pub source: DimRef<B>,
    pub kind: ExtentKind,
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

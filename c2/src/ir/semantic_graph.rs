use super::common::{ExprId, Op, TensorType, ValueId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub inputs: Vec<TensorType>,
    pub stages: Vec<Stage>,
    pub outputs: Vec<ValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub value: ValueId,
    pub expr: ExprId,
    pub op: Op,
    pub inputs: Vec<Use>,
    pub axes: Vec<Axis>,
    pub output: Index,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub value: ValueId,
    pub index: Index,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Axis {
    pub extent: LocalExtentSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Index(pub Vec<usize>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LocalExtentSource {
    Const(usize),
    InputDim { input: usize, dim: usize },
}

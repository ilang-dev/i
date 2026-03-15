use super::common::{Extent, Op, Pattern, TensorType, ValueId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub inputs: Vec<TensorType>,
    pub stages: Vec<Stage>,
    pub outputs: Vec<ValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub op: Op,
    pub inputs: Vec<Use>,
    pub axes: Vec<Axis>,
    pub output: Pattern,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub value: ValueId,
    pub pattern: Pattern,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Axis {
    pub name: char,
    pub extent: Extent,
}

use super::common::{Extent, Op, Pattern, Shape, ValueId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub values: Vec<Value>,
    pub inputs: Vec<ValueId>,
    pub stages: Vec<Stage>,
    pub outputs: Vec<ValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Value {
    pub shape: Shape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub op: Op,
    pub inputs: Vec<Use>,
    pub output: ValueId,
    pub axes: Vec<Axis>,
    pub output_pattern: Pattern,
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

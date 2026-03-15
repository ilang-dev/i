use super::common::{AffineMap, Axis, AxisId, StageId, TensorType, ValueId};
use super::component::Operator;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub inputs: Vec<Value>,
    pub stages: Vec<Stage>,
    pub outputs: Vec<ValueId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Value {
    pub id: ValueId,
    pub ty: TensorType,
    pub kind: ValueKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValueKind {
    Input { index: usize },
    StageResult { stage: StageId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub id: StageId,
    pub op: Operator,
    pub inputs: Vec<Use>,
    pub output: ValueId,
    pub domain: Domain,
    pub output_map: AffineMap,
    pub reduction: Vec<AxisId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Use {
    pub value: ValueId,
    pub access: AffineMap,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Domain {
    pub axes: Vec<Axis>,
}

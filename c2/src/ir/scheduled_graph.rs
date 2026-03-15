use super::common::{AxisId, Extent, FusionId, StageId, ValueId};
use super::semantic_graph;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub semantic: semantic_graph::Graph,
    pub stage_schedules: Vec<StageSchedule>,
    pub fusion_groups: Vec<FusionGroup>,
    pub storage: Vec<StoragePlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageSchedule {
    pub stage: StageId,
    pub splits: Vec<Split>,
    pub order: Vec<LoopAxis>,
    pub compute_at: Placement,
    pub store_at: Placement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Split {
    pub axis: AxisId,
    pub factors: Vec<Extent>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LoopAxis {
    pub axis: AxisId,
    pub part: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Placement {
    Root,
    At { stage: StageId, axis: LoopAxis },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionGroup {
    pub id: FusionId,
    pub stages: Vec<StageId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoragePlan {
    pub value: ValueId,
    pub materialization: Materialization,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Materialization {
    Inline,
    Buffer { at: Placement, folded: bool },
}

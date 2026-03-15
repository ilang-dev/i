use super::common::{ExprId, LoopVar, Split};
use super::semantic_graph;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub semantic: semantic_graph::Graph,
    pub schedule: ScheduleIntent,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScheduleIntent {
    pub exprs: Vec<ExprSchedule>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExprSchedule {
    pub expr: ExprId,
    pub splits: Vec<Split>,
    pub order: Vec<LoopVar>,
    pub compute_at: Vec<Option<LoopVar>>,
}

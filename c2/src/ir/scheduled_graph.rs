use super::common::{LoopVar, Split, StageId};
use super::semantic_graph;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub semantic: semantic_graph::Graph,
    pub stages: Vec<StageSchedule>,
    pub fusions: Vec<Vec<StageId>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageSchedule {
    pub splits: Vec<Split>,
    pub order: Vec<LoopVar>,
    pub compute_at: Site,
    pub store_at: Option<Site>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Site {
    Root,
    At(StageId, LoopVar),
}

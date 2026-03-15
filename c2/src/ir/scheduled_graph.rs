use super::common::{Extent, StageId};
use super::semantic_graph;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph {
    pub semantic: semantic_graph::Graph,
    pub stages: Vec<StageSchedule>,
    pub fusions: Vec<Fusion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageSchedule {
    pub splits: Vec<Split>,
    pub order: Vec<Loop>,
    pub compute_at: Site,
    pub store_at: Site,
    pub materialize: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Split {
    pub axis: char,
    pub factors: Vec<Extent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Loop {
    pub axis: char,
    pub part: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Site {
    Root,
    Loop { stage: StageId, loop_: Loop },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fusion {
    pub stages: Vec<StageId>,
}

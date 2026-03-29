//! Scheduled stage IR.
//!
//! This module defines the payload of `Graph<ScheduledStage>`.
//! `Stage` gives the semantic shape of one node.
//! `Schedule` gives the loop structure of that node.
//!
//! Invariants:
//! - `Stage` axes are local and canonical.
//! - `Index` values refer only to `Stage` axes.
//! - `Schedule.splits.len() == stage.rank`.
//! - `AxisRef { part: 0 }` names the base loop of an axis.
//! - An axis with `n` split factors has parts `0..=n`.
//! - `Schedule.order` contains each loop part exactly once.
//! - `Schedule.compute_sites.len() == stage.inputs.len()`.
//! - `Schedule.init_site` is `Some(site)` for a reduction stage and `None` for
//!   a pointwise stage.
//!
use super::common::Op;

/// One scheduled stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledStage {
    pub stage: Stage,
    pub schedule: Schedule,
}

/// Semantic content of one 𝚒 expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    /// Scalar operator applied at each point in the domain.
    pub op: Op,
    /// Number of axes in the stage domain.
    pub rank: usize,
    /// One explicit access pattern per input, in input order.
    pub inputs: Vec<Index>,
    /// Explicit output access pattern.
    pub output: Index,
}

/// Indexing of one tensor access by stage axes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Index(pub Vec<Axis>);

/// One stage axis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Axis(pub usize);

/// Schedule of one stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Schedule {
    pub splits: Vec<SplitList>,
    pub order: Vec<AxisRef>,
    pub compute_sites: Vec<Option<Site>>,
    pub init_site: Option<Site>,
}

/// Split factors of one stage axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitList(pub Vec<SplitFactor>);

/// One split factor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitFactor(pub usize);

/// One site in a stage loop nest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Site {
    Root,
    At(AxisRef),
}

/// One loop part of one stage axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisRef {
    pub axis: Axis,
    pub part: usize,
}

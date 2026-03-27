//! Kernel IR.
//!
//! This module defines one kernel as an ordered dataflow graph over scheduled
//! stages.
//! `Kernel` wraps `Graph<ScheduledStage>`.
//!
//! Invariants:
//! - `Kernel.0.inputs` is ordered.
//! - `Kernel.0.nodes` is ordered.
//! - `Kernel.0.outputs` is ordered.
//! - Every `Kernel.0` node has exactly one output.
//! - A node input sourced from `Input(_)` is produced outside the kernel.
//! - A node input sourced from `Node(_, _)` is produced inside the kernel.
//!
use super::graph::Graph;
use super::stage::ScheduledStage;

/// One kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel(pub Graph<ScheduledStage>);

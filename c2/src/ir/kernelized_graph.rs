//! KernelizedGraph IR.
//!
//! This module defines one kernelized graph as an ordered dataflow graph over
//! scheduled stages.
//! `KernelizedGraph` wraps `Graph<ScheduledStage>`.
//!
//! Invariants:
//! - `KernelizedGraph.0.inputs` is ordered.
//! - `KernelizedGraph.0.nodes` is ordered.
//! - `KernelizedGraph.0.outputs` is ordered.
//! - Every `KernelizedGraph.0` node has exactly one output.
//! - A node input sourced from `Input(_)` is produced outside the kernelized graph.
//! - A node input sourced from `Node(_, _)` is produced inside the kernelized graph.
//!
use super::graph::Graph;
use super::stage::ScheduledStage;

/// One kernelized graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelizedGraph(pub Graph<ScheduledStage>);

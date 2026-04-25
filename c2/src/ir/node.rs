//! Node IR.
//!
//! This module defines the payload of `Graph<Node>`.
//! `Node` gives the semantic index shape of one graph node.
//! `MultiIndex` values tie inputs and outputs to those indexes.
//!
//! Invariants:
//! - `Node` indexes are local and canonical.
//! - `Index` values refer only to `Node` indexes.
//! - `Node.splits.len() == node.rank`.
//! - `AxisRef { level: 0 }` names the base loop of an index.
//! - An index with `n` split factors has levels `0..=n`.
//! - `Node.order` contains each loop level of each index exactly once.
//! - `Node.compute_sites.len() == node.inputs.len()`.
//! - `Node.init_site` is `Some(site)` for a reduction node and `None` for
//!   a pointwise node.
//!

use super::common::{Index, Op};

// TODO: a `shape` can be determined for this node, but not without the full
// graph (`shape` is input-valued). should it be included here?
/// One graph node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    /// Scalar operator applied at each point in the domain.
    pub op: Op,
    /// Number of indexes in the node domain.
    pub rank: usize,
    /// One explicit access pattern per input, in input order.
    pub inputs: Vec<MultiIndex>,
    /// Explicit output access pattern.
    pub output: MultiIndex,
    /// Split factors for each node index.
    pub splits: Vec<SplitList>,
    /// Loop order of one node.
    pub order: Vec<AxisRef>,
    /// One compute site per input, in input order.
    pub compute_sites: Vec<Option<Site>>,
    /// Output init site for reductions.
    pub init_site: Option<Site>,
}

/// Indexing of one tensor access by node indexes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiIndex(pub Vec<Index>);

/// Split factors of one node index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitList(pub Vec<SplitFactor>);

/// One split factor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitFactor(pub usize);

/// One site in a node loop nest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Site {
    Root,
    At(AxisRef),
}

/// One loop level of one node index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisRef {
    pub index: Index,
    pub level: usize,
}

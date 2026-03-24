//! Graph IR.
//!
//! This module defines an ordered dataflow graph over node payloads.
//! `Graph<T>` gives ordered inputs, ordered nodes, and ordered outputs.
//!
//! Invariants:
//! - Graph boundaries are explicit.
//! - `Graph.inputs` is ordered.
//! - `Graph.nodes` is ordered.
//! - `Graph.outputs` is ordered.
//! - `InputId(i)` names `Graph.inputs[i]`.
//! - `NodeId(i)` names `Graph.nodes[i]`.
//! - `Node.inputs` is in operand order.
//! - `Graph.outputs` is in user-visible order.
//!
/// One ordered dataflow graph over payloads of type `T`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph<T> {
    pub inputs: Vec<Input>,
    pub nodes: Vec<Node<T>>,
    pub outputs: Vec<Source>,
}

/// One interior graph node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node<T> {
    pub inner: T,
    pub inputs: Vec<Source>,
}

/// One ordered graph input.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Input;

/// Handle for one graph input.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InputId(pub usize);

/// Handle for one interior graph node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub usize);

/// Source of a node input or graph output.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Source {
    Input(InputId),
    Node(NodeId),
}

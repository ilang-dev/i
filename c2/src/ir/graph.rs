/// Ordered dataflow graph over node payloads of type `T`.
///
/// Graph boundaries are explicit and ordered:
/// `inputs` gives the leaf tensors of the whole program or subgraph,
/// and `outputs` gives the roots in user-visible order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Graph<T> {
    pub inputs: Vec<Input>,
    pub nodes: Vec<Node<T>>,
    pub outputs: Vec<Source>,
}

/// One interior node of a [`Graph`].
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

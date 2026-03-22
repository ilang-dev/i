pub use super::expr::{Expr, PermutationAtom};

/// Source-shaped program structure for 𝚒.
///
/// A `Component` is a tree of combinators whose leaves are atomic source
/// expressions. It preserves how a program is assembled before combinators
/// are resolved into explicit dataflow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Component {
    /// One atomic 𝚒 expression.
    Expr(Expr),
    /// Wires the leaves of the left component to the roots of the right.
    ///
    /// Mixed arity is handled left-to-right. Any unpaired leaves or roots are
    /// left in place.
    Compose(Box<Component>, Box<Component>),
    /// Wires the roots of the left component to the leaves of the right.
    ///
    /// Mixed arity is handled left-to-right. Any unpaired roots or leaves are
    /// left in place.
    Chain(Box<Component>, Box<Component>),
    /// Merges inputs pairwise so both components can read the same sources.
    ///
    /// Mixed arity is handled left-to-right. Any unpaired inputs remain as
    /// independent inputs of the combined component.
    Fanout(Box<Component>, Box<Component>),
    /// Concatenates the inputs and outputs of two components.
    Pair(Box<Component>, Box<Component>),
    /// Swaps the first two outputs of one component.
    Swap(Box<Component>),
}

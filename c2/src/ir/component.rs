//! Component IR.
//!
//! This module defines the source-shaped program tree of 𝚒.
//! `Component` combines expression leaves with combinator nodes.
//!
//! Invariants:
//! - Every leaf is an 𝚒 expression (`Expr` variant).
//! - Every interior node is a combinator (non-`Expr` variant).
//! - Child order is explicit and preserved.
//! - Combinator structure is explicit and preserved.
//!
pub use super::expr::{Expr, PermutationAtom};

/// One component tree.
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

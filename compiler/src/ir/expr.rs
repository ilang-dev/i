//! Source expression IR.
//!
//! This module defines the parsed local form of one 𝚒 expression.
//! `Expr` records the operator, index patterns, split directives, and
//! permutation stream of that expression.
//!
//! Invariants:
//! - `Expr.inputs` preserves source input order.
//! - Each input pattern preserves source index order.
//! - `Expr.output` preserves source index order.
//! - `Expr.splits` preserves source split order.
//! - `Expr.permutation` preserves source permutation order.
//! - Axis names are expression-local source chars.
//! - `PermutationAtom::Input(operand)` names one operand of the same `Expr`.
//!
use super::common::Op;

/// One parsed 𝚒 expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expr {
    /// Scalar operator applied by this expression.
    pub op: Op,
    /// Input index patterns in source order.
    pub inputs: Vec<Vec<char>>,
    /// Output index pattern.
    pub output: Vec<char>,
    /// Source split directives in source order.
    ///
    /// Each entry gives the axis being split together with the user-written
    /// factor list for that split.
    pub splits: Vec<(char, Vec<usize>)>,
    /// Source permutation stream.
    ///
    /// Axis atoms describe the loop stack. Input atoms mark the depth at
    /// which a given input is computed within that stack.
    pub permutation: Vec<PermutationAtom>,
}

/// One token in an expression permutation stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PermutationAtom {
    /// A loop over one axis part.
    Axis { axis: char, part: usize },
    /// A marker indicating where one operand is computed.
    Input(Operand),
    /// A marker indicating where the output is initialized.
    Bang,
}

/// One input operand of an expression.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Operand {
    /// Left operand.
    Left,
    /// Right operand.
    Right,
}

use super::common::Op;

/// Parsed source form of one scheduled 𝚒 expression.
///
/// This type preserves the local structure written by the user: the operator,
/// input and output index patterns, split directives, and the permutation
/// stream that describes loop order together with input compute placement
/// markers.
///
/// `Expr` is intentionally source-shaped. It is not a semantic stage and it
/// does not resolve dataflow beyond what is written inside the expression
/// itself.
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

/// One token in the source permutation stream of an [`Expr`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PermutationAtom {
    /// A loop over one axis part.
    Axis { axis: char, part: usize },
    /// A marker indicating where one input is computed.
    Input(usize),
}

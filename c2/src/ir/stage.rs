use super::common::Op;

/// Expression-local semantic content of one 𝚒 expression.
///
/// A `Stage` consists of a scalar op, a local iteration rank, one explicit
/// index per input, and one explicit index for the output.
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

/// One axis of the domain.
///
/// `Axis(n)` denotes the `n`th axis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Axis(pub usize);

/// Explicit indexing of one tensor access by `Stage` axes.
///
/// Each entry specifies the stage axis used for one tensor dimension in tensor
/// dimension order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Index(pub Vec<Axis>);

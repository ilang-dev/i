use super::common::Op;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledStage {
    stage: Stage,
    schedule: Schedule,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Schedule {
    splits: Vec<SplitList>, // assert!(splits.len() == stage.rank)
    order: Vec<AxisRef>, // must represent every loop exactly once, where part 0 is the base loop and valid parts for axis a are 0..=splits[a].0.len(); assert!(order.len() == splits.iter().map(|x| x.0.len() + 1).sum())
    compute_sites: Vec<Option<Site>>, // per input, assert!(compute_sites.len() == stage.inputs.len())
    init_site: Option<Site>, // reduction init site; for pointwise stages assert!(init_site.is_none())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitList(Vec<SplitFactor>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitFactor(usize);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Site {
    Root,
    At(AxisRef),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisRef {
    pub axis: Axis,
    pub part: usize,
}

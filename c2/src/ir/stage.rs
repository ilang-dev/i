//! Stage IR.
//!
//! This module defines the payload of `Graph<Stage>`.
//! `Stage` gives the physical axis shape of one graph node.
//! `AxisMultiId` values tie inputs and outputs to those axes.
//!
//! Invariants:
//! - `Stage.axes` is ordered.
//! - `AxisId(i)` names `Stage.axes[i]`.
//! - `AxisMultiId` values refer only to `Stage` axes.
//! - `AxisMultiId` values preserve access order.
//! - `Axis::Live` carries one semantic index source and one extent kind.
//! - `Axis::Pruned` is resolved through graph edges by aligning output axes
//!   with consumer input axes.
//! - `Output.axes` gives the stage output layout.
//! - `Input.axes` gives one stage input layout.
//! - `Output.init` is `Some(site)` for a reduction stage and `None` for a
//!   pointwise stage.
//!

use super::common::{ExtentKind, Index, Op};

/// One physical stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    /// Scalar operator applied at each point in the domain.
    pub op: Op,
    /// Physical axes of the stage domain.
    pub axes: Vec<Axis>,
    /// Explicit output access pattern.
    pub output: Output,
    /// One explicit access pattern per input, in input order.
    pub inputs: Vec<Input>,
}

/// One physical axis slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    /// One live physical axis.
    Live {
        /// Semantic index source of this axis.
        index: Index,
        /// Physical extent kind of this axis.
        kind: ExtentKind,
    },
    /// One pruned physical axis.
    Pruned,
}

/// Handle for one physical axis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AxisId(pub usize);

/// Indexing of one tensor access by physical axes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisMultiId(pub Vec<AxisId>);

/// One stage output access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Output {
    /// Output layout in physical axes.
    pub axes: AxisMultiId,
    /// Output init site.
    pub init: Option<Site>,
}

/// One stage input access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Input {
    /// Input layout in physical axes.
    pub axes: AxisMultiId,
    /// Input compute site.
    pub compute: Option<Site>,
}

/// One site in a stage axis nest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Site {
    /// The root of the stage.
    Root,
    /// The body of one physical axis.
    At(AxisId),
}

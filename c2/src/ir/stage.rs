//! Stage IR.
//!
//! This module defines the payload of `Graph<Stage>`.
//! `Stage` gives the physical axes of one graph node.
//! `Shape` values give local semantic buffer dimensions.
//! `Layout` values give input and output buffer dimensions.
//!
//! Invariants:
//! - `Stage.axes` is ordered.
//! - `AxisId(i)` names `Stage.axes[i]`.
//! - `AxisRef::Local(axis)` names one axis of the same `Stage`.
//! - `AxisRef::Consumer(axis)` names one axis of the consuming `Stage`.
//! - `AxisRef::Consumer(_)` appears only in stages placed under a consumer.
//! - `Axis::Live` carries one semantic index source and one extent kind.
//! - `Axis::Pruned` carries the consumer axis supplying that axis.
//! - Every pruned axis is resolved through graph edges by aligning output
//!   shape with consumer input shape.
//! - `Shape` values preserve semantic dimension order.
//! - `Shape` values refer only to semantic indexes of the same `Stage`.
//! - `Layout` values preserve tensor dimension order.
//! - `LayoutDim::Physical` names one dimension sourced by one physical axis.
//! - `LayoutDim::Semantic` names one dimension sourced by one semantic shape
//!   dimension.
//! - `LayoutDim::Semantic.axes` is ordered by physical extent kind for that
//!   semantic index.
//! - `Output.shape` gives the stage output shape.
//! - `Output.layout` gives the stage output layout.
//! - `Input.shape` gives one stage input shape.
//! - `Input.layout` gives one stage input layout.
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
    /// Explicit output buffer.
    pub output: Output,
    /// One explicit input buffer per input, in input order.
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
    Pruned {
        /// Consumer axis supplying this axis.
        by: AxisRef,
    },
}

/// Handle for one stage axis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AxisId(pub usize);

/// Reference to one available axis iterator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisRef {
    /// One local stage axis.
    Local(AxisId),
    /// One consumer axis supplying a pruned stage axis.
    Consumer(AxisId),
}

/// One tensor layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout(pub Vec<LayoutDim>);

/// One tensor shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape(pub Vec<Index>);

/// One dimension of one tensor layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutDim {
    /// One dimension sourced by one physical axis.
    Physical(AxisRef),
    /// One dimension sourced by one semantic shape dimension.
    Semantic {
        /// Semantic index naming the shape dimension.
        index: Index,
        /// Axis iterators composing the semantic iterator.
        axes: Vec<AxisRef>,
    },
}

/// One stage output buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Output {
    /// Output shape.
    pub shape: Shape,
    /// Output layout.
    pub layout: Layout,
    /// Output init site.
    pub init: Option<Site>,
}

/// One stage input buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Input {
    /// Input shape.
    pub shape: Shape,
    /// Input layout.
    pub layout: Layout,
    /// Input compute site.
    pub compute: Option<Site>,
}

/// One site in a stage axis nest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Site {
    /// The root of the stage.
    Root,
    /// The body of one local stage axis.
    At(AxisId),
}

//! Kernel program IR.
//!
//! This module defines planned kernel dataflow.
//! `KernelProgram` gives logical buffers and an ordered graph of kernels.
//! `Kernel` gives one kernel body over one buffer reference domain.
//! `BufferShape` values give semantic buffer dimensions.
//! `BufferLayout` values give physical buffer dimensions.
//! `Block` values give ordered kernel statements.
//! `Extent` values give loop bounds in terms of logical buffer dimensions.
//! `Access` values give logical buffer indexing.
//!
//! Invariants:
//! - `KernelProgram.buffers` is ordered.
//! - `KernelProgram.outputs` is ordered.
//! - `KernelProgram.graph` is ordered.
//! - `BufferId(i)` names `KernelProgram.buffers[i]`.
//! - `BufferId` values are graph-local.
//! - `KernelProgram.outputs` gives program output buffers in user-visible
//!   order.
//! - `Buffer.kind` gives the graph boundary role of one logical buffer.
//! - `Buffer.shape` gives the semantic shape of one logical buffer.
//! - `Buffer.layout` gives the allocation layout of one logical buffer.
//! - `BufferShape` preserves semantic buffer dimension order.
//! - `BufferLayout` preserves physical buffer dimension order.
//! - `Kernel<B>` is parameterized by buffer reference type.
//! - `Kernel.reads` is the ordered `readonlys` parameter list.
//! - `Kernel.writes` is the ordered `writeables` parameter list.
//! - Every buffer accessed by a kernel appears in `Kernel.reads` or
//!   `Kernel.writes`.
//! - `Action::Init.write` names one buffer in `Kernel.writes`.
//! - `Action::Compute.write` names one buffer in `Kernel.writes`.
//! - `LoopId` values are kernel-local.
//! - `LoopId(i)` names one loop in the same `Kernel`.
//! - `LoopId` values are unique within one `Kernel`.
//! - `Block` statements execute in order.
//! - `Action::Loop` contains one nested loop body.
//! - `Action::Init` initializes one scalar element of one write buffer.
//! - `Action::Compute` computes one scalar element of one write buffer.
//! - `Extent<B>.source` names the semantic dimension supplying the loop bound.
//! - `Extent<B>.kind` gives the physical extent kind of the loop.
//! - `TailGuard(true)` requests the canonical tail guard for a loop.
//! - `DimRef<B> { buffer, dim }` names dimension `dim` of `buffer`.
//! - `Access<B>.index` preserves buffer layout dimension order.
//! - `Iter::Raw(loop_id)` uses one loop iterator directly.
//! - `Iter::Reconstructed` composes one semantic iterator from physical loop
//!   iterators.
//! - `Iter::Reconstructed.loops` is ordered by physical extent kind.
//! - `Iter::Reconstructed.factors` is ordered by split level.
//!

use super::common::{ExtentKind, Op};
use super::graph::Graph;

/// One kernel-level program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelProgram {
    /// Logical buffers in graph order.
    pub buffers: Vec<Buffer>,
    /// Program output buffers in user-visible order.
    pub outputs: Vec<BufferId>,
    /// Kernel dataflow graph.
    pub graph: Graph<Kernel>,
}

/// One planned kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel<B = BufferId> {
    /// Kernel `readonlys` parameter buffers.
    pub reads: Vec<B>,
    /// Kernel `writeables` parameter buffers.
    pub writes: Vec<B>,
    /// Kernel body.
    pub body: Block<B>,
}

/// One logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    /// Buffer role.
    pub kind: BufferKind,
    /// Buffer semantic shape.
    pub shape: BufferShape,
    /// Buffer allocation layout.
    pub layout: BufferLayout,
}

/// Role of one logical buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferKind {
    /// One program input buffer.
    Input,
    /// One intermediate buffer.
    Intermediate,
    /// One program output buffer.
    Output,
}

/// Semantic shape of one logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferShape(pub Vec<DimRef<BufferId>>);

/// Physical allocation layout of one logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferLayout(pub Vec<Extent<BufferId>>);

/// One ordered block of kernel statements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block<B = BufferId>(pub Vec<Action<B>>);

/// One kernel statement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action<B = BufferId> {
    /// One counted loop.
    Loop {
        /// Loop identifier.
        id: LoopId,
        /// Loop extent.
        extent: Extent<B>,
        /// Loop tail guard.
        guard: TailGuard,
        /// Loop body.
        body: Block<B>,
    },
    /// One scalar initialization.
    Init {
        /// Reduction operator being initialized.
        op: Op,
        /// Write access being initialized.
        write: Access<B>,
    },
    /// One scalar computation.
    Compute {
        /// Scalar operator being applied.
        op: Op,
        /// Write access being computed.
        write: Access<B>,
        /// Read accesses used by the computation.
        reads: Vec<Access<B>>,
    },
}

/// Handle for one logical buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BufferId(pub usize);

/// Handle for one kernel loop.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoopId(pub usize);

/// Whether a loop has the canonical tail guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TailGuard(pub bool);

/// One loop extent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Extent<B = BufferId> {
    /// Semantic dimension supplying this extent.
    pub source: DimRef<B>,
    /// Physical extent kind.
    pub kind: ExtentKind,
}

/// Reference to one dimension of one buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DimRef<B = BufferId> {
    /// Buffer.
    pub buffer: B,
    /// Buffer dimension.
    pub dim: usize,
}

/// One indexed buffer access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Access<B = BufferId> {
    /// Buffer being accessed.
    pub buffer: B,
    /// Index expression for each layout dimension.
    pub index: Vec<Iter>,
}

/// One iterator expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Iter {
    /// One raw loop iterator.
    Raw(LoopId),
    /// One semantic iterator reconstructed from physical loop iterators.
    Reconstructed {
        /// Physical loop iterators.
        loops: Vec<LoopId>,
        /// Split factors.
        factors: Vec<usize>,
    },
}

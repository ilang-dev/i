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
//! - `Buffer.scope` gives the storage scope of one logical buffer.
//! - `Buffer.shape` gives the semantic shape of one logical buffer.
//! - `Buffer.layout` gives the allocation layout of one logical buffer.
//! - `BufferShape` preserves semantic buffer dimension order.
//! - `BufferLayout` preserves physical buffer dimension order.
//! - `Kernel<B, L>` is parameterized by buffer reference type and loop mode
//!   type.
//! - `Kernel.reads` is the ordered `readonlys` parameter list.
//! - `Kernel.writes` is the ordered `writeables` parameter list.
//! - Every buffer accessed by a kernel appears in `Kernel.reads` or
//!   `Kernel.writes`.
//! - `Action::Init.write` names one buffer in `Kernel.writes`.
//! - `Action::Init.zero_checks` names reduction loops exterior to the init.
//! - `Action::Compute.write` names one buffer in `Kernel.writes`.
//! - `LoopId` values are kernel-local.
//! - `LoopId(i)` names one loop in the same `Kernel`.
//! - `LoopId` values are unique within one `Kernel`.
//! - `LoopMode::Serial` requires counted-loop execution.
//! - `LoopMode::Parallel` permits target parallel binding.
//! - `Block` statements execute in order.
//! - `Action::Loop` contains one nested loop body.
//! - `Action::Init` initializes one scalar element of one write buffer.
//! - `Action::Init` executes iff every loop in `zero_checks` is zero.
//! - Empty `Action::Init.zero_checks` gives unconditional initialization.
//! - `Action::Compute` computes one scalar element of one write buffer.
//! - `Action::Snapshot` copies one scalar element from a kernel buffer to a
//!   write buffer.
//! - `Action::Scale` multiplies one scalar accumulator element by a scalar
//!   factor expression.
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

use super::common::{DimRef, Extent, Op};
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
pub struct Kernel<B = BufferId, L = LoopMode> {
    /// Kernel `readonlys` parameter buffers.
    pub reads: Vec<B>,
    /// Kernel `writeables` parameter buffers.
    pub writes: Vec<B>,
    /// Kernel body.
    pub body: Block<B, L>,
}

/// One logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    /// Buffer role.
    pub kind: BufferKind,
    /// Buffer storage scope.
    pub scope: BufferScope,
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

/// Storage scope of one logical buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferScope {
    /// One exec-owned allocation.
    Global,
    /// One cooperative-group allocation.
    Group,
    /// One execution-lane allocation.
    Private,
}

/// Semantic shape of one logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferShape(pub Vec<DimRef<BufferId>>);

/// Physical allocation layout of one logical buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferLayout(pub Vec<Extent<BufferId>>);

/// One ordered block of kernel statements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block<B = BufferId, L = LoopMode>(pub Vec<Action<B, L>>);

/// One kernel statement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action<B = BufferId, L = LoopMode> {
    /// One counted loop.
    Loop {
        /// Loop identifier.
        id: LoopId,
        /// Loop execution mode.
        mode: L,
        /// Loop extent.
        extent: Extent<B>,
        /// Loop tail guard.
        guard: TailGuard,
        /// Loop body.
        body: Block<B, L>,
    },
    /// One scalar initialization.
    Init {
        /// Reduction operator being initialized.
        op: Op,
        /// Write access being initialized.
        write: Access<B>,
        /// Reduction loops that must be zero.
        zero_checks: Vec<LoopId>,
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
    /// One scalar snapshot.
    Snapshot {
        /// Snapshot destination.
        write: Access<B>,
        /// Snapshot source.
        read: Access<B>,
    },
    /// One scalar accumulator scale correction.
    Scale {
        /// Accumulator access being scaled.
        write: Access<B>,
        /// Multiplicative scale factor.
        factor: ScalarExpr<B>,
    },
}

/// One scalar expression usable as a scale factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScalarExpr<B = BufferId> {
    /// One buffer access.
    Access(Access<B>),
    /// One unary scalar operation.
    Unary {
        /// Unary scalar operator.
        op: Op,
        /// Operand expression.
        arg: Box<ScalarExpr<B>>,
    },
    /// One binary scalar operation.
    Binary {
        /// Binary scalar operator.
        op: Op,
        /// Left operand expression.
        lhs: Box<ScalarExpr<B>>,
        /// Right operand expression.
        rhs: Box<ScalarExpr<B>>,
    },
}

/// Handle for one logical buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BufferId(pub usize);

/// Handle for one kernel loop.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoopId(pub usize);

/// Execution mode of one kernel loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopMode {
    /// Counted-loop execution.
    Serial,
    /// Target parallel binding permitted.
    Parallel,
}

/// Whether a loop has the canonical tail guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TailGuard(pub bool);

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

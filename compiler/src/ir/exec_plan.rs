//! Exec plan IR.
//!
//! This module defines the generated public execution plan of 𝚒.
//! `ExecPlan` gives kernels, runtime buffers, public output
//! metadata, and ordered execution steps.
//! `Shape` values give semantic dimensions in terms of program inputs.
//! `Layout` values give physical allocation dimensions in terms of program
//! inputs.
//! `BufferRef` values name runtime buffers at exec call sites.
//!
//! Invariants:
//! - `ExecPlan.kernels` is ordered.
//! - `ExecPlan.buffers.inputs` is ordered.
//! - `ExecPlan.buffers.intermediates` is ordered.
//! - `ExecPlan.buffers.outputs` is ordered.
//! - `ExecPlan.count == ExecPlan.ranks.len()`.
//! - `ExecPlan.count == ExecPlan.shapes.len()`.
//! - `ExecPlan.count == ExecPlan.buffers.outputs.len()`.
//! - `KernelId(i)` names `ExecPlan.kernels[i]`.
//! - `Input(i)` names `ExecPlan.buffers.inputs[i]`.
//! - `Intermediate(i)` names `ExecPlan.buffers.intermediates[i]`.
//! - `Output(i)` names `ExecPlan.buffers.outputs[i]`.
//! - `BoundKernel.reads` is the ordered `readonlys` parameter list.
//! - `BoundKernel.writes` is the ordered `writeables` parameter list.
//! - Every value in `BoundKernel.reads` has `Arg::Readonly`.
//! - Every value in `BoundKernel.writes` has `Arg::Writeable`.
//! - `BoundKernel.locals` is ordered.
//! - `LoopBind::Serial` gives counted-loop execution.
//! - `LoopBind::Group` binds one loop to a cooperative-group dimension.
//! - `LoopBind::Lane` binds one loop to an execution-lane dimension.
//! - `Param { arg, ind }` names parameter `ind` of `arg`.
//! - `Local(i)` names `BoundKernel.locals[i]`.
//! - `KernelRef::Param(param)` names one kernel parameter.
//! - `KernelRef::Local(local)` names one kernel-local buffer.
//! - `Shape` preserves semantic dimension order.
//! - `Layout` preserves physical allocation dimension order.
//! - `Shape` and `Layout` dimensions are sourced from program inputs.
//! - `BufferRef` names one runtime buffer.
//! - `Exec` steps execute in order.
//! - `Step::Alloc` allocates one intermediate buffer.
//! - `Step::Dispatch` calls one kernel.
//! - Each kernel is dispatched exactly once.
//! - `Step::Dispatch.reads` is ordered as kernel `readonlys`.
//! - `Step::Dispatch.writes` is ordered as kernel `writeables`.
//! - `Step::Free` releases one intermediate buffer.
//!

use super::common::{DimRef, Extent};
use super::kernel_program::{Block, BufferScope};

/// One public execution plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecPlan {
    /// Kernels in execution-plan order.
    pub kernels: Vec<BoundKernel>,
    /// Runtime buffers.
    pub buffers: Buffers,
    /// Number of program outputs.
    pub count: usize,
    /// Rank of each program output.
    pub ranks: Vec<usize>,
    /// Shape of each program output.
    pub shapes: Vec<Shape>,
    /// Ordered execution steps.
    pub exec: Exec,
}

/// Runtime buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffers {
    /// Program input buffers.
    pub inputs: Vec<Buffer>,
    /// Exec-owned intermediate buffers.
    pub intermediates: Vec<Buffer>,
    /// Program output buffers.
    pub outputs: Vec<Buffer>,
}

/// One execution-plan buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Buffer {
    /// Buffer semantic shape.
    pub shape: Shape,
    /// Buffer allocation layout.
    pub layout: Layout,
}

/// One bound kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundKernel {
    /// Kernel `readonlys` parameter list.
    pub reads: Vec<Param>,
    /// Kernel `writeables` parameter list.
    pub writes: Vec<Param>,
    /// Kernel-local buffers.
    pub locals: Vec<LocalBuffer>,
    /// Kernel body.
    pub body: Block<KernelRef, LoopBind>,
}

/// One kernel-local buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalBuffer {
    /// Buffer storage scope.
    pub scope: BufferScope,
    /// Buffer metadata.
    pub buffer: Buffer,
}

/// Semantic shape of one runtime buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape(pub Vec<DimRef<Input>>);

/// Physical allocation layout of one runtime buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout(pub Vec<Extent<Input>>);

/// Reference to one runtime buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferRef {
    /// One program input buffer.
    Input(Input),
    /// One exec-owned intermediate buffer.
    Intermediate(Intermediate),
    /// One program output buffer.
    Output(Output),
}

/// Handle for one program input buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Input(pub usize);

/// Handle for one exec-owned intermediate buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Intermediate(pub usize);

/// Handle for one program output buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Output(pub usize);

/// Kernel ABI argument bucket.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Arg {
    /// The `readonlys` argument.
    Readonly,
    /// The `writeables` argument.
    Writeable,
}

/// Handle for one kernel ABI parameter.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Param {
    /// ABI argument bucket.
    pub arg: Arg,
    /// Parameter index within the bucket.
    pub ind: usize,
}

/// Reference to one kernel buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelRef {
    /// One kernel ABI parameter.
    Param(Param),
    /// One kernel-local buffer.
    Local(Local),
}

/// Handle for one kernel-local buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Local(pub usize);

/// Execution binding of one kernel loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopBind {
    /// Counted-loop execution.
    Serial,
    /// Cooperative-group dimension.
    Group {
        /// Dimension number.
        dim: usize,
    },
    /// Execution-lane dimension.
    Lane {
        /// Dimension number.
        dim: usize,
    },
}

/// Ordered execution steps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Exec(pub Vec<Step>);

/// One execution step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    /// Allocate one intermediate buffer.
    Alloc(Intermediate),
    /// Dispatch one kernel.
    Dispatch {
        /// Kernel being dispatched.
        kernel: KernelId,
        /// Runtime buffers bound to `readonlys`.
        reads: Vec<BufferRef>,
        /// Runtime buffers bound to `writeables`.
        writes: Vec<BufferRef>,
    },
    /// Free one intermediate buffer.
    Free(Intermediate),
}

/// Handle for one planned kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelId(pub usize);

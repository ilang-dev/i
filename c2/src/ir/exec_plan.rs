//! Exec plan IR.
//!
//! This module defines the generated public execution plan of 𝚒.
//! `ExecPlan` gives param-bound kernels, runtime buffers, public output
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
//! - `Kernel<Param>.reads` is the ordered `readonlys` parameter list.
//! - `Kernel<Param>.writes` is the ordered `writeables` parameter list.
//! - Every value in `Kernel<Param>.reads` has `Arg::Readonly`.
//! - Every value in `Kernel<Param>.writes` has `Arg::Writeable`.
//! - `Param { arg, ind }` names parameter `ind` of `arg`.
//! - `Shape` preserves semantic dimension order.
//! - `Layout` preserves physical allocation dimension order.
//! - `Shape` and `Layout` dimensions are sourced from program inputs.
//! - `BufferRef` names one runtime buffer.
//! - `Exec` steps execute in order.
//! - `Step::Alloc` allocates one intermediate buffer.
//! - `Step::Dispatch` calls one kernel.
//! - `Step::Dispatch.reads` is ordered as kernel `readonlys`.
//! - `Step::Dispatch.writes` is ordered as kernel `writeables`.
//! - `Step::Free` releases one intermediate buffer.
//!

use super::kernel_program::{DimRef, Extent, Kernel};

/// One public execution plan.
struct ExecPlan {
    /// Param-bound kernels in execution-plan order.
    kernels: Vec<Kernel<Param>>,
    /// Runtime buffers.
    buffers: Buffers,
    /// Number of program outputs.
    count: usize,
    /// Rank of each program output.
    ranks: Vec<usize>,
    /// Shape of each program output.
    shapes: Vec<Shape>,
    /// Ordered execution steps.
    exec: Exec,
}

/// Runtime buffers.
struct Buffers {
    /// Program input buffers.
    inputs: Vec<InputBuffer>,
    /// Exec-owned intermediate buffers.
    intermediates: Vec<IntermediateBuffer>,
    /// Program output buffers.
    outputs: Vec<OutputBuffer>,
}

/// One program input buffer.
struct InputBuffer {
    /// Buffer semantic shape.
    shape: Shape,
    /// Buffer allocation layout.
    layout: Layout,
}

/// One exec-owned intermediate buffer.
struct IntermediateBuffer {
    /// Buffer semantic shape.
    shape: Shape,
    /// Buffer allocation layout.
    layout: Layout,
}

/// One program output buffer.
struct OutputBuffer {
    /// Buffer semantic shape.
    shape: Shape,
    /// Buffer allocation layout.
    layout: Layout,
}

/// Semantic shape of one runtime buffer.
struct Shape(Vec<DimRef<Input>>);

/// Physical allocation layout of one runtime buffer.
struct Layout(Vec<Extent<Input>>);

/// Reference to one runtime buffer.
enum BufferRef {
    /// One program input buffer.
    Input(Input),
    /// One exec-owned intermediate buffer.
    Intermediate(Intermediate),
    /// One program output buffer.
    Output(Output),
}

/// Handle for one program input buffer.
struct Input(usize);

/// Handle for one exec-owned intermediate buffer.
struct Intermediate(usize);

/// Handle for one program output buffer.
struct Output(usize);

/// Kernel ABI argument bucket.
enum Arg {
    /// The `readonlys` argument.
    Readonly,
    /// The `writeables` argument.
    Writeable,
}

/// Handle for one kernel ABI parameter.
struct Param {
    /// ABI argument bucket.
    arg: Arg,
    /// Parameter index within the bucket.
    ind: usize,
}

/// Ordered execution steps.
struct Exec(Vec<Step>);

/// One execution step.
enum Step {
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
struct KernelId(usize);

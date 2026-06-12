//! Parallel module IR.
//!
//! This module defines the generated parallel execution module of 𝚒.
//! `ParallelModule` gives device kernels, runtime buffers, public output
//! metadata, and ordered host execution steps.
//! `DeviceKernel` gives one kernel body in one execution context.
//! `ExecutionShape` gives the group and lane dimensions of a device launch.
//! `HostStep::Launch` binds runtime buffers to one device kernel.
//!
//! Invariants:
//! - `ParallelModule.kernels` is ordered.
//! - `ParallelModule.buffers.inputs` is ordered.
//! - `ParallelModule.buffers.intermediates` is ordered.
//! - `ParallelModule.buffers.outputs` is ordered.
//! - `ParallelModule.count == ParallelModule.ranks.len()`.
//! - `ParallelModule.count == ParallelModule.shapes.len()`.
//! - `ParallelModule.count == ParallelModule.buffers.outputs.len()`.
//! - `DeviceKernelId(i)` names `ParallelModule.kernels[i]`.
//! - `Input(i)` names `ParallelModule.buffers.inputs[i]`.
//! - `Intermediate(i)` names `ParallelModule.buffers.intermediates[i]`.
//! - `Output(i)` names `ParallelModule.buffers.outputs[i]`.
//! - `DeviceKernel.reads` is the ordered read-only parameter list.
//! - `DeviceKernel.writes` is the ordered writable parameter list.
//! - Every value in `DeviceKernel.reads` has `Arg::Readonly`.
//! - Every value in `DeviceKernel.writes` has `Arg::Writeable`.
//! - `DeviceKernel.locals` is ordered.
//! - `DeviceKernel.execution` gives replicated execution contexts.
//! - `ExecutionShape.groups` is ordered by group dimension.
//! - `ExecutionShape.lanes` is ordered by lane dimension.
//! - `ExecutionDim.id` names one execution-space index.
//! - `Param { arg, ind }` names parameter `ind` of `arg`.
//! - `Local(i)` names `DeviceKernel.locals[i]`.
//! - `KernelRef::Param(param)` names one device-kernel parameter.
//! - `KernelRef::Local(local)` names one device-kernel local buffer.
//! - `Shape` preserves semantic dimension order.
//! - `Layout` preserves physical allocation dimension order.
//! - `Shape` and `Layout` dimensions are sourced from program inputs.
//! - `BufferRef` names one runtime buffer.
//! - `HostExec` steps execute in order.
//! - `HostStep::Alloc` allocates one intermediate buffer.
//! - `HostStep::Launch` launches one device kernel.
//! - Each device kernel is launched exactly once.
//! - `HostStep::Launch.reads` is ordered as kernel `reads`.
//! - `HostStep::Launch.writes` is ordered as kernel `writes`.
//! - `HostStep::Free` releases one intermediate buffer.
//!

use super::exec_plan::{
    BufferRef, Buffers, ExecutionShape, Intermediate, KernelRef, LocalBuffer, Param, Shape,
};
use super::kernel_program::Block;

/// One parallel execution module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelModule {
    /// Device kernels in module order.
    pub kernels: Vec<DeviceKernel>,
    /// Runtime buffers.
    pub buffers: Buffers,
    /// Number of program outputs.
    pub count: usize,
    /// Rank of each program output.
    pub ranks: Vec<usize>,
    /// Shape of each program output.
    pub shapes: Vec<Shape>,
    /// Ordered host execution steps.
    pub exec: HostExec,
}

/// One device kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceKernel {
    /// Read-only parameter list.
    pub reads: Vec<Param>,
    /// Writable parameter list.
    pub writes: Vec<Param>,
    /// Device-kernel-local buffers.
    pub locals: Vec<LocalBuffer>,
    /// Device execution-space shape.
    pub execution: ExecutionShape,
    /// Device-kernel body.
    pub body: Block<KernelRef>,
}

/// Ordered host execution steps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostExec(pub Vec<HostStep>);

/// One host execution step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostStep {
    /// Allocate one intermediate buffer.
    Alloc(Intermediate),
    /// Launch one device kernel.
    Launch {
        /// Device kernel being launched.
        kernel: DeviceKernelId,
        /// Runtime buffers bound to read-only parameters.
        reads: Vec<BufferRef>,
        /// Runtime buffers bound to writable parameters.
        writes: Vec<BufferRef>,
    },
    /// Free one intermediate buffer.
    Free(Intermediate),
}

/// Handle for one device kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceKernelId(pub usize);

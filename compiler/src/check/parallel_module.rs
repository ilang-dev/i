use std::fmt;

use crate::check::exec_plan::validate_exec_plan;
use crate::ir::exec_plan::{BoundKernel, Exec, ExecPlan, KernelId, Step};
use crate::ir::parallel_module::{HostStep, ParallelModule};

pub fn validate_parallel_module(module: &ParallelModule) -> Result<(), ValidationError> {
    validate_exec_plan(&to_exec_plan(module)).map_err(ValidationError::from_exec_plan)
}

fn to_exec_plan(module: &ParallelModule) -> ExecPlan {
    ExecPlan {
        kernels: module
            .kernels
            .iter()
            .map(|kernel| BoundKernel {
                reads: kernel.reads.clone(),
                writes: kernel.writes.clone(),
                locals: kernel.locals.clone(),
                execution: kernel.execution.clone(),
                body: kernel.body.clone(),
            })
            .collect(),
        buffers: module.buffers.clone(),
        count: module.count,
        ranks: module.ranks.clone(),
        shapes: module.shapes.clone(),
        exec: Exec(
            module
                .exec
                .0
                .iter()
                .map(|step| match step {
                    HostStep::Alloc(intermediate) => Step::Alloc(*intermediate),
                    HostStep::Launch {
                        kernel,
                        reads,
                        writes,
                    } => Step::Dispatch {
                        kernel: KernelId(kernel.0),
                        reads: reads.clone(),
                        writes: writes.clone(),
                    },
                    HostStep::Free(intermediate) => Step::Free(*intermediate),
                })
                .collect(),
        ),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub message: String,
}

impl ValidationError {
    fn from_exec_plan(error: crate::check::exec_plan::ValidationError) -> Self {
        let message = error
            .message
            .replace("dispatch", "launch")
            .replace("kernel", "device kernel");
        Self { message }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::validate_parallel_module;
    use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
    use crate::ir::exec_plan::{
        Arg, Buffer, BufferRef, Buffers, ExecutionDim, ExecutionShape, Input, Intermediate,
        KernelRef, Layout, Param, Shape,
    };
    use crate::ir::kernel_program::{Access, Action, Block, LoopId, TailGuard};
    use crate::ir::parallel_module::{
        DeviceKernel, DeviceKernelId, HostExec, HostStep, ParallelModule,
    };

    fn valid_module() -> ParallelModule {
        ParallelModule {
            kernels: vec![DeviceKernel {
                reads: vec![Param {
                    arg: Arg::Readonly,
                    ind: 0,
                }],
                writes: vec![Param {
                    arg: Arg::Writeable,
                    ind: 0,
                }],
                locals: vec![],
                execution: ExecutionShape {
                    groups: vec![],
                    lanes: vec![ExecutionDim {
                        id: LoopId(0),
                        extent: Extent {
                            source: DimRef {
                                buffer: KernelRef::Param(Param {
                                    arg: Arg::Readonly,
                                    ind: 0,
                                }),
                                dim: 0,
                            },
                            kind: ExtentKind::Semantic,
                        },
                        guard: TailGuard(false),
                    }],
                },
                body: Block(vec![Action::Compute {
                    op: Op::Add,
                    write: Access {
                        buffer: KernelRef::Param(Param {
                            arg: Arg::Writeable,
                            ind: 0,
                        }),
                        index: vec![crate::ir::kernel_program::Iter::Raw(LoopId(0))],
                    },
                    reads: vec![Access {
                        buffer: KernelRef::Param(Param {
                            arg: Arg::Readonly,
                            ind: 0,
                        }),
                        index: vec![crate::ir::kernel_program::Iter::Raw(LoopId(0))],
                    }],
                }]),
            }],
            buffers: Buffers {
                inputs: vec![vector_buffer()],
                intermediates: vec![vector_buffer()],
                outputs: vec![vector_buffer()],
            },
            count: 1,
            ranks: vec![1],
            shapes: vec![Shape(vec![DimRef {
                buffer: Input(0),
                dim: 0,
            }])],
            exec: HostExec(vec![
                HostStep::Alloc(Intermediate(0)),
                HostStep::Launch {
                    kernel: DeviceKernelId(0),
                    reads: vec![BufferRef::Input(Input(0))],
                    writes: vec![BufferRef::Output(crate::ir::exec_plan::Output(0))],
                },
                HostStep::Free(Intermediate(0)),
            ]),
        }
    }

    fn vector_buffer() -> Buffer {
        Buffer {
            shape: Shape(vec![DimRef {
                buffer: Input(0),
                dim: 0,
            }]),
            layout: Layout(vec![Extent {
                source: DimRef {
                    buffer: Input(0),
                    dim: 0,
                },
                kind: ExtentKind::Semantic,
            }]),
        }
    }

    #[test]
    fn accepts_parallel_module() {
        validate_parallel_module(&valid_module()).unwrap();
    }

    #[test]
    fn rejects_unknown_device_kernel() {
        let mut module = valid_module();
        let HostStep::Launch { kernel, .. } = &mut module.exec.0[1] else {
            unreachable!();
        };
        *kernel = DeviceKernelId(1);

        assert!(validate_parallel_module(&module)
            .unwrap_err()
            .message
            .contains("references nonexistent device kernel 1"));
    }

    #[test]
    fn rejects_too_many_lane_dimensions() {
        let mut module = valid_module();
        let extent = module.kernels[0].execution.lanes[0].extent.clone();
        module.kernels[0].execution.lanes.push(ExecutionDim {
            id: LoopId(1),
            extent: extent.clone(),
            guard: TailGuard(false),
        });
        module.kernels[0].execution.lanes.push(ExecutionDim {
            id: LoopId(2),
            extent: extent.clone(),
            guard: TailGuard(false),
        });
        module.kernels[0].execution.lanes.push(ExecutionDim {
            id: LoopId(3),
            extent,
            guard: TailGuard(false),
        });

        assert!(validate_parallel_module(&module)
            .unwrap_err()
            .message
            .contains("has 4 lane dims"));
    }
}

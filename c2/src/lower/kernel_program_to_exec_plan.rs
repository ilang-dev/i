use std::collections::BTreeSet;
use std::fmt;

use crate::check::exec_plan::validate_exec_plan;
use crate::check::kernel_program::validate_kernel_program;
use crate::ir::exec_plan::{
    Arg, BufferRef, Buffers, Exec, ExecPlan, Input, InputBuffer, Intermediate, IntermediateBuffer,
    KernelId, Layout, Output, OutputBuffer, Param, Shape, Step,
};
use crate::ir::kernel_program::{
    Access, Action, Block, BufferId, BufferKind, DimRef, Extent, Kernel, KernelProgram,
};

pub fn lower_kernel_program_to_exec_plan(program: &KernelProgram) -> Result<ExecPlan, LowerError> {
    validate_kernel_program(program).map_err(LowerError::from_kernel_program)?;
    let builder = Builder::new(program)?;
    let plan = builder.lower()?;
    validate_exec_plan(&plan).map_err(LowerError::from_exec_plan)?;
    Ok(plan)
}

struct Builder<'a> {
    program: &'a KernelProgram,
    buffer_refs: Vec<Option<BufferRef>>,
}

impl<'a> Builder<'a> {
    fn new(program: &'a KernelProgram) -> Result<Self, LowerError> {
        let mut builder = Self {
            program,
            buffer_refs: vec![None; program.buffers.len()],
        };
        builder.assign_buffer_refs()?;
        Ok(builder)
    }

    fn lower(&self) -> Result<ExecPlan, LowerError> {
        let buffers = self.lower_buffers()?;
        let shapes = buffers
            .outputs
            .iter()
            .map(|buffer| buffer.shape.clone())
            .collect::<Vec<_>>();
        let ranks = shapes.iter().map(|shape| shape.0.len()).collect::<Vec<_>>();
        let kernels = self.lower_kernels()?;
        let exec = self.lower_exec()?;

        Ok(ExecPlan {
            count: shapes.len(),
            kernels,
            buffers,
            ranks,
            shapes,
            exec,
        })
    }

    fn assign_buffer_refs(&mut self) -> Result<(), LowerError> {
        let mut next_input = 0;
        let mut next_intermediate = 0;

        for (buffer_index, buffer) in self.program.buffers.iter().enumerate() {
            match buffer.kind {
                BufferKind::Input => {
                    self.buffer_refs[buffer_index] = Some(BufferRef::Input(Input(next_input)));
                    next_input += 1;
                }
                BufferKind::Intermediate => {
                    self.buffer_refs[buffer_index] =
                        Some(BufferRef::Intermediate(Intermediate(next_intermediate)));
                    next_intermediate += 1;
                }
                BufferKind::Output => {}
            }
        }

        let mut outputs = BTreeSet::new();
        for (output_index, buffer) in self.program.outputs.iter().enumerate() {
            let Some(kind) = self.program.buffers.get(buffer.0).map(|buffer| buffer.kind) else {
                return Err(LowerError::new(format!(
                    "output {} references nonexistent buffer {}",
                    output_index, buffer.0
                )));
            };
            if kind != BufferKind::Output {
                return Err(LowerError::new(format!(
                    "output {} references {:?} buffer {}",
                    output_index, kind, buffer.0
                )));
            }
            if !outputs.insert(buffer.0) {
                return Err(LowerError::new(format!(
                    "output {} repeats buffer {}",
                    output_index, buffer.0
                )));
            }
            self.buffer_refs[buffer.0] = Some(BufferRef::Output(Output(output_index)));
        }

        for (buffer_index, buffer) in self.program.buffers.iter().enumerate() {
            if buffer.kind == BufferKind::Output && !outputs.contains(&buffer_index) {
                return Err(LowerError::new(format!(
                    "output buffer {} is not a program output",
                    buffer_index
                )));
            }
        }

        Ok(())
    }

    fn lower_buffers(&self) -> Result<Buffers, LowerError> {
        let mut inputs = Vec::new();
        let mut intermediates = Vec::new();
        let mut outputs = vec![None; self.program.outputs.len()];

        for (buffer_index, buffer) in self.program.buffers.iter().enumerate() {
            match self.buffer_ref(BufferId(buffer_index))? {
                BufferRef::Input(_) => inputs.push(InputBuffer {
                    shape: self.lower_shape(BufferId(buffer_index))?,
                    layout: self.lower_layout(BufferId(buffer_index))?,
                }),
                BufferRef::Intermediate(_) => intermediates.push(IntermediateBuffer {
                    shape: self.lower_shape(BufferId(buffer_index))?,
                    layout: self.lower_layout(BufferId(buffer_index))?,
                }),
                BufferRef::Output(output) => {
                    outputs[output.0] = Some(OutputBuffer {
                        shape: self.lower_shape(BufferId(buffer_index))?,
                        layout: self.lower_layout(BufferId(buffer_index))?,
                    });
                }
            }

            if buffer.shape.0.is_empty() && !buffer.layout.0.is_empty() {
                return Err(LowerError::new(format!(
                    "buffer {} has scalar shape and non-scalar layout",
                    buffer_index
                )));
            }
        }

        let outputs = outputs
            .into_iter()
            .enumerate()
            .map(|(output, buffer)| {
                buffer.ok_or_else(|| LowerError::new(format!("output {} was not lowered", output)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Buffers {
            inputs,
            intermediates,
            outputs,
        })
    }

    fn lower_shape(&self, buffer: BufferId) -> Result<Shape, LowerError> {
        let shape = &self.program.buffers[buffer.0].shape;
        shape
            .0
            .iter()
            .enumerate()
            .map(|(dim, source)| {
                self.resolve_shape_dim(*source, &mut Vec::new())
                    .map_err(|error| {
                        LowerError::new(format!(
                            "buffer {} shape dim {} {}",
                            buffer.0, dim, error.message
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Shape)
    }

    fn lower_layout(&self, buffer: BufferId) -> Result<Layout, LowerError> {
        let layout = &self.program.buffers[buffer.0].layout;
        layout
            .0
            .iter()
            .enumerate()
            .map(|(dim, extent)| {
                Ok(Extent {
                    source: self
                        .resolve_shape_dim(extent.source, &mut Vec::new())
                        .map_err(|error| {
                            LowerError::new(format!(
                                "buffer {} layout dim {} {}",
                                buffer.0, dim, error.message
                            ))
                        })?,
                    kind: extent.kind.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Layout)
    }

    fn resolve_shape_dim(
        &self,
        dim: DimRef<BufferId>,
        stack: &mut Vec<BufferId>,
    ) -> Result<DimRef<Input>, LowerError> {
        let buffer = self.program.buffers.get(dim.buffer.0).ok_or_else(|| {
            LowerError::new(format!("references nonexistent buffer {}", dim.buffer.0))
        })?;

        if dim.dim >= buffer.shape.0.len() {
            return Err(LowerError::new(format!(
                "references nonexistent dim {} of buffer {}",
                dim.dim, dim.buffer.0
            )));
        }

        if let BufferRef::Input(input) = self.buffer_ref(dim.buffer)? {
            return Ok(DimRef {
                buffer: input,
                dim: dim.dim,
            });
        }

        if stack.contains(&dim.buffer) {
            return Err(LowerError::new(format!(
                "shape dim cycle through buffer {}",
                dim.buffer.0
            )));
        }

        stack.push(dim.buffer);
        let source = buffer.shape.0[dim.dim];
        let resolved = self.resolve_shape_dim(source, stack);
        stack.pop();
        resolved
    }

    fn lower_kernels(&self) -> Result<Vec<Kernel<Param>>, LowerError> {
        self.program
            .graph
            .nodes
            .iter()
            .enumerate()
            .map(|(kernel_index, node)| self.lower_kernel(kernel_index, &node.inner))
            .collect()
    }

    fn lower_kernel(
        &self,
        kernel_index: usize,
        kernel: &Kernel<BufferId>,
    ) -> Result<Kernel<Param>, LowerError> {
        Ok(Kernel {
            reads: (0..kernel.reads.len())
                .map(|ind| Param {
                    arg: Arg::Readonly,
                    ind,
                })
                .collect(),
            writes: (0..kernel.writes.len())
                .map(|ind| Param {
                    arg: Arg::Writeable,
                    ind,
                })
                .collect(),
            body: self.lower_block(kernel, &kernel.body).map_err(|error| {
                LowerError::new(format!("kernel {} {}", kernel_index, error.message))
            })?,
        })
    }

    fn lower_block(
        &self,
        kernel: &Kernel<BufferId>,
        block: &Block<BufferId>,
    ) -> Result<Block<Param>, LowerError> {
        block
            .0
            .iter()
            .map(|action| self.lower_action(kernel, action))
            .collect::<Result<Vec<_>, _>>()
            .map(Block)
    }

    fn lower_action(
        &self,
        kernel: &Kernel<BufferId>,
        action: &Action<BufferId>,
    ) -> Result<Action<Param>, LowerError> {
        match action {
            Action::Loop {
                id,
                extent,
                guard,
                body,
            } => Ok(Action::Loop {
                id: *id,
                extent: Extent {
                    source: DimRef {
                        buffer: self.lower_param(kernel, extent.source.buffer)?,
                        dim: extent.source.dim,
                    },
                    kind: extent.kind.clone(),
                },
                guard: *guard,
                body: self.lower_block(kernel, body)?,
            }),
            Action::Init { op, write } => Ok(Action::Init {
                op: *op,
                write: self.lower_access(kernel, write)?,
            }),
            Action::Compute { op, write, reads } => Ok(Action::Compute {
                op: *op,
                write: self.lower_access(kernel, write)?,
                reads: reads
                    .iter()
                    .map(|access| self.lower_access(kernel, access))
                    .collect::<Result<Vec<_>, _>>()?,
            }),
        }
    }

    fn lower_access(
        &self,
        kernel: &Kernel<BufferId>,
        access: &Access<BufferId>,
    ) -> Result<Access<Param>, LowerError> {
        Ok(Access {
            buffer: self.lower_param(kernel, access.buffer)?,
            index: access.index.clone(),
        })
    }

    fn lower_param(
        &self,
        kernel: &Kernel<BufferId>,
        buffer: BufferId,
    ) -> Result<Param, LowerError> {
        if let Some(ind) = kernel
            .reads
            .iter()
            .position(|candidate| *candidate == buffer)
        {
            return Ok(Param {
                arg: Arg::Readonly,
                ind,
            });
        }
        if let Some(ind) = kernel
            .writes
            .iter()
            .position(|candidate| *candidate == buffer)
        {
            return Ok(Param {
                arg: Arg::Writeable,
                ind,
            });
        }
        Err(LowerError::new(format!(
            "buffer {} is not a kernel parameter",
            buffer.0
        )))
    }

    fn lower_exec(&self) -> Result<Exec, LowerError> {
        let mut steps = Vec::new();
        for intermediate in 0..self.count_intermediates() {
            steps.push(Step::Alloc(Intermediate(intermediate)));
        }
        for (kernel_index, node) in self.program.graph.nodes.iter().enumerate() {
            steps.push(Step::Dispatch {
                kernel: KernelId(kernel_index),
                reads: node
                    .inner
                    .reads
                    .iter()
                    .copied()
                    .map(|buffer| self.buffer_ref(buffer))
                    .collect::<Result<Vec<_>, _>>()?,
                writes: node
                    .inner
                    .writes
                    .iter()
                    .copied()
                    .map(|buffer| self.buffer_ref(buffer))
                    .collect::<Result<Vec<_>, _>>()?,
            });
        }
        for intermediate in (0..self.count_intermediates()).rev() {
            steps.push(Step::Free(Intermediate(intermediate)));
        }
        Ok(Exec(steps))
    }

    fn count_intermediates(&self) -> usize {
        self.program
            .buffers
            .iter()
            .filter(|buffer| buffer.kind == BufferKind::Intermediate)
            .count()
    }

    fn buffer_ref(&self, buffer: BufferId) -> Result<BufferRef, LowerError> {
        self.buffer_refs
            .get(buffer.0)
            .copied()
            .flatten()
            .ok_or_else(|| LowerError::new(format!("buffer {} has no exec ref", buffer.0)))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LowerError {
    pub message: String,
}

impl LowerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn from_kernel_program(error: crate::check::kernel_program::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_exec_plan(error: crate::check::exec_plan::ValidationError) -> Self {
        Self::new(error.to_string())
    }
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LowerError {}

#[cfg(test)]
mod tests {
    use super::lower_kernel_program_to_exec_plan;
    use crate::front::parse_expr;
    use crate::ir::common::ExtentKind;
    use crate::ir::exec_plan::{Arg, BufferRef, Input, Intermediate, Output, Param, Step};
    use crate::ir::graph::{Graph, Input as GraphInput};
    use crate::ir::kernel_program::{
        Action, Block, Buffer, BufferId, BufferKind, BufferLayout, BufferShape, DimRef, Extent,
        KernelProgram,
    };
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;
    use crate::{component, front};

    fn lower_expr(src: &str) -> crate::ir::exec_plan::ExecPlan {
        let component = component::expr(parse_expr(src).unwrap());
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        lower_kernel_program_to_exec_plan(&kernel_program).unwrap()
    }

    fn lower_component(
        component: &crate::ir::component::Component,
    ) -> crate::ir::exec_plan::ExecPlan {
        let graph = lower_component_to_graph(component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        lower_kernel_program_to_exec_plan(&kernel_program).unwrap()
    }

    #[test]
    fn lowers_single_kernel_program() {
        let plan = lower_expr("ik*kj~ijk");

        assert_eq!(plan.kernels.len(), 1);
        assert_eq!(
            plan.kernels[0].reads,
            vec![
                Param {
                    arg: Arg::Readonly,
                    ind: 0
                },
                Param {
                    arg: Arg::Readonly,
                    ind: 1
                }
            ]
        );
        assert_eq!(
            plan.kernels[0].writes,
            vec![Param {
                arg: Arg::Writeable,
                ind: 0
            }]
        );
        assert_eq!(plan.buffers.inputs.len(), 2);
        assert_eq!(plan.buffers.intermediates.len(), 0);
        assert_eq!(plan.buffers.outputs.len(), 1);
        assert_eq!(plan.count, 1);
        assert_eq!(plan.ranks, vec![3]);
        assert_eq!(
            plan.exec.0,
            vec![Step::Dispatch {
                kernel: crate::ir::exec_plan::KernelId(0),
                reads: vec![BufferRef::Input(Input(0)), BufferRef::Input(Input(1))],
                writes: vec![BufferRef::Output(Output(0))]
            }]
        );
    }

    #[test]
    fn lowers_chain_with_intermediate_allocation() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let plan = lower_component(&component);

        assert_eq!(plan.kernels.len(), 2);
        assert_eq!(plan.buffers.intermediates.len(), 1);
        assert_eq!(
            plan.exec.0,
            vec![
                Step::Alloc(Intermediate(0)),
                Step::Dispatch {
                    kernel: crate::ir::exec_plan::KernelId(0),
                    reads: vec![BufferRef::Input(Input(0)), BufferRef::Input(Input(1))],
                    writes: vec![BufferRef::Intermediate(Intermediate(0))]
                },
                Step::Dispatch {
                    kernel: crate::ir::exec_plan::KernelId(1),
                    reads: vec![BufferRef::Intermediate(Intermediate(0))],
                    writes: vec![BufferRef::Output(Output(0))]
                },
                Step::Free(Intermediate(0)),
            ]
        );
    }

    #[test]
    fn resolves_public_output_shape_to_input_dims() {
        let plan = lower_expr("+ij~ji");

        assert_eq!(
            plan.shapes,
            vec![crate::ir::exec_plan::Shape(vec![
                crate::ir::kernel_program::DimRef {
                    buffer: Input(0),
                    dim: 1
                },
                crate::ir::kernel_program::DimRef {
                    buffer: Input(0),
                    dim: 0
                },
            ])]
        );
        assert_eq!(plan.ranks, vec![2]);
    }

    #[test]
    fn resolves_layout_sources_to_input_dims() {
        let plan = lower_kernel_program_to_exec_plan(&layout_resolution_program()).unwrap();
        let output = &plan.buffers.outputs[0];

        assert_eq!(
            output.layout,
            crate::ir::exec_plan::Layout(vec![
                Extent {
                    source: crate::ir::kernel_program::DimRef {
                        buffer: Input(0),
                        dim: 1
                    },
                    kind: ExtentKind::Base(vec![4])
                },
                Extent {
                    source: crate::ir::kernel_program::DimRef {
                        buffer: Input(0),
                        dim: 1
                    },
                    kind: ExtentKind::Split {
                        level: 1,
                        factor: 4
                    }
                },
                Extent {
                    source: crate::ir::kernel_program::DimRef {
                        buffer: Input(0),
                        dim: 0
                    },
                    kind: ExtentKind::Semantic
                },
            ])
        );
    }

    #[test]
    fn binds_kernel_body_accesses_to_params() {
        let plan = lower_expr("+ij~ji");
        let action = first_compute(&plan.kernels[0].body);

        let Action::Compute { write, reads, .. } = action else {
            unreachable!();
        };
        assert_eq!(
            write.buffer,
            Param {
                arg: Arg::Writeable,
                ind: 0
            }
        );
        assert_eq!(
            reads[0].buffer,
            Param {
                arg: Arg::Readonly,
                ind: 0
            }
        );
    }

    #[test]
    fn rejects_program_output_that_is_not_output_buffer() {
        let mut program = scalar_passthrough_program(BufferKind::Input);

        let error = lower_kernel_program_to_exec_plan(&program).unwrap_err();
        assert_eq!(error.to_string(), "output 0 references Input buffer 0");

        program.buffers[0].kind = BufferKind::Intermediate;
        let error = lower_kernel_program_to_exec_plan(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "output 0 references Intermediate buffer 0"
        );
    }

    #[test]
    fn rejects_output_buffer_missing_from_program_outputs() {
        let mut program = scalar_passthrough_program(BufferKind::Output);
        program.outputs = vec![];

        let error = lower_kernel_program_to_exec_plan(&program).unwrap_err();
        assert_eq!(error.to_string(), "output buffer 0 is not a program output");
    }

    #[test]
    fn rejects_shape_dim_cycle() {
        let program = scalar_passthrough_program(BufferKind::Output);

        let error = lower_kernel_program_to_exec_plan(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "buffer 0 shape dim 0 shape dim cycle through buffer 0"
        );
    }

    fn scalar_passthrough_program(kind: BufferKind) -> KernelProgram {
        KernelProgram {
            buffers: vec![Buffer {
                kind,
                shape: BufferShape(vec![DimRef {
                    buffer: BufferId(0),
                    dim: 0,
                }]),
                layout: BufferLayout(vec![Extent {
                    source: DimRef {
                        buffer: BufferId(0),
                        dim: 0,
                    },
                    kind: ExtentKind::Semantic,
                }]),
            }],
            outputs: vec![BufferId(0)],
            graph: Graph {
                inputs: vec![GraphInput],
                nodes: vec![],
                outputs: vec![],
            },
        }
    }

    fn layout_resolution_program() -> KernelProgram {
        KernelProgram {
            buffers: vec![
                Buffer {
                    kind: BufferKind::Input,
                    shape: BufferShape(vec![
                        DimRef {
                            buffer: BufferId(0),
                            dim: 0,
                        },
                        DimRef {
                            buffer: BufferId(0),
                            dim: 1,
                        },
                    ]),
                    layout: BufferLayout(vec![
                        Extent {
                            source: DimRef {
                                buffer: BufferId(0),
                                dim: 0,
                            },
                            kind: ExtentKind::Semantic,
                        },
                        Extent {
                            source: DimRef {
                                buffer: BufferId(0),
                                dim: 1,
                            },
                            kind: ExtentKind::Semantic,
                        },
                    ]),
                },
                Buffer {
                    kind: BufferKind::Output,
                    shape: BufferShape(vec![
                        DimRef {
                            buffer: BufferId(0),
                            dim: 1,
                        },
                        DimRef {
                            buffer: BufferId(0),
                            dim: 0,
                        },
                    ]),
                    layout: BufferLayout(vec![
                        Extent {
                            source: DimRef {
                                buffer: BufferId(1),
                                dim: 0,
                            },
                            kind: ExtentKind::Base(vec![4]),
                        },
                        Extent {
                            source: DimRef {
                                buffer: BufferId(1),
                                dim: 0,
                            },
                            kind: ExtentKind::Split {
                                level: 1,
                                factor: 4,
                            },
                        },
                        Extent {
                            source: DimRef {
                                buffer: BufferId(1),
                                dim: 1,
                            },
                            kind: ExtentKind::Semantic,
                        },
                    ]),
                },
            ],
            outputs: vec![BufferId(1)],
            graph: Graph {
                inputs: vec![GraphInput],
                nodes: vec![],
                outputs: vec![],
            },
        }
    }

    fn first_compute(block: &Block<Param>) -> &Action<Param> {
        for action in &block.0 {
            match action {
                Action::Compute { .. } => return action,
                Action::Loop { body, .. } => {
                    return first_compute(body);
                }
                Action::Init { .. } => {}
            }
        }
        unreachable!()
    }
}

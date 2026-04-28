use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::kernel_program::validate_kernel_program;
use crate::check::stage::validate_stage_graph;
use crate::ir::common::{DimRef, Extent, ExtentKind};
use crate::ir::graph::{
    Graph, InputId, Node as GraphNode, NodeId, Output as GraphOutput, OutputId, Source,
};
use crate::ir::kernel_program::{
    Access, Action, Block, Buffer, BufferId, BufferKind, BufferLayout, BufferShape, Iter, Kernel,
    KernelProgram, LoopId, TailGuard,
};
use crate::ir::stage::{
    Axis, AxisId, AxisRef as StageAxisRef, Layout, LayoutDim, ProgramShape, ShapeDim, Site, Stage,
    StageProgram,
};

pub fn lower_stage_program_to_kernel_program(
    program: &StageProgram,
) -> Result<KernelProgram, LowerError> {
    validate_stage_graph(&program.graph).map_err(LowerError::from_stage_graph)?;
    let builder = Builder::new(program)?;
    let kernel_program = builder.lower()?;
    validate_kernel_program(&kernel_program).map_err(LowerError::from_kernel_program)?;
    Ok(kernel_program)
}

struct Builder<'a> {
    program: &'a StageProgram,
    buffers: Vec<Buffer>,
    input_buffers: Vec<BufferId>,
    stage_buffers: Vec<BufferId>,
    buffer_sources: Vec<Option<Source>>,
    kernel_nodes: Vec<GraphNode<Kernel>>,
}

impl<'a> Builder<'a> {
    fn new(program: &'a StageProgram) -> Result<Self, LowerError> {
        if program.shapes.inputs.len() != program.graph.inputs.len() {
            return Err(LowerError::new(format!(
                "stage program has {} input shapes for {} graph inputs",
                program.shapes.inputs.len(),
                program.graph.inputs.len()
            )));
        }
        if program.shapes.nodes.len() != program.graph.nodes.len() {
            return Err(LowerError::new(format!(
                "stage program has {} node shapes for {} graph nodes",
                program.shapes.nodes.len(),
                program.graph.nodes.len()
            )));
        }

        let mut builder = Self {
            program,
            buffers: Vec::new(),
            input_buffers: Vec::new(),
            stage_buffers: Vec::new(),
            buffer_sources: Vec::new(),
            kernel_nodes: Vec::new(),
        };
        builder.build_input_buffers();
        builder.build_stage_buffers()?;
        Ok(builder)
    }

    fn lower(mut self) -> Result<KernelProgram, LowerError> {
        let outputs = self.lower_outputs()?;
        for stage_index in 0..self.program.graph.nodes.len() {
            if self.is_consumer_scoped_stage(stage_index) {
                continue;
            }
            self.lower_kernel(stage_index)?;
        }
        let graph_outputs = outputs
            .iter()
            .copied()
            .map(|buffer| {
                self.buffer_sources[buffer.0].ok_or_else(|| {
                    LowerError::new(format!(
                        "program output buffer {} has no graph source",
                        buffer.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(KernelProgram {
            buffers: self.buffers,
            outputs,
            graph: Graph {
                inputs: self.program.graph.inputs.clone(),
                nodes: self.kernel_nodes,
                outputs: graph_outputs,
            },
        })
    }

    fn build_input_buffers(&mut self) {
        for (input, shape) in self.program.shapes.inputs.iter().enumerate() {
            let buffer = BufferId(self.buffers.len());
            self.input_buffers.push(buffer);
            self.buffer_sources
                .push(Some(Source::Input(InputId(input))));
            self.buffers.push(Buffer {
                kind: BufferKind::Input,
                shape: self.lower_program_shape(shape),
                layout: BufferLayout(
                    (0..shape.0.len())
                        .map(|dim| Extent {
                            source: DimRef { buffer, dim },
                            kind: ExtentKind::Semantic,
                        })
                        .collect(),
                ),
            });
        }
    }

    fn build_stage_buffers(&mut self) -> Result<(), LowerError> {
        let output_stages = self
            .program
            .graph
            .outputs
            .iter()
            .filter_map(|source| match source {
                Source::Node(node, OutputId(0)) => Some(node.0),
                _ => None,
            })
            .collect::<BTreeSet<_>>();

        for (stage_index, node) in self.program.graph.nodes.iter().enumerate() {
            let buffer = BufferId(self.buffers.len());
            self.stage_buffers.push(buffer);
            self.buffer_sources.push(None);
            let kind = if output_stages.contains(&stage_index) {
                BufferKind::Output
            } else {
                BufferKind::Intermediate
            };
            self.buffers.push(Buffer {
                kind,
                shape: self.lower_program_shape(&self.program.shapes.nodes[stage_index]),
                layout: self.lower_output_buffer_layout(stage_index, buffer, &node.inner)?,
            });
        }

        Ok(())
    }

    fn lower_program_shape(&self, shape: &ProgramShape) -> BufferShape {
        BufferShape(
            shape
                .0
                .iter()
                .map(|dim| self.lower_shape_dim(*dim))
                .collect(),
        )
    }

    fn lower_shape_dim(&self, dim: ShapeDim) -> DimRef<BufferId> {
        DimRef {
            buffer: self.input_buffers[dim.input.0],
            dim: dim.dim,
        }
    }

    fn lower_output_buffer_layout(
        &self,
        stage_index: usize,
        buffer: BufferId,
        stage: &Stage,
    ) -> Result<BufferLayout, LowerError> {
        let extents = stage
            .output
            .layout
            .0
            .iter()
            .map(|dim| match dim {
                LayoutDim::Physical(axis) => {
                    let axis = self.stage_axis(stage, *axis)?;
                    Ok(Extent {
                        source: self.stage_output_dim(stage, buffer, axis.index)?,
                        kind: axis.kind.clone(),
                    })
                }
                LayoutDim::Semantic { index, .. } => Ok(Extent {
                    source: self.stage_output_dim(stage, buffer, *index)?,
                    kind: ExtentKind::Semantic,
                }),
            })
            .collect::<Result<Vec<_>, LowerError>>()
            .map_err(|error| {
                LowerError::new(format!(
                    "stage {} output layout {}",
                    stage_index, error.message
                ))
            })?;
        Ok(BufferLayout(extents))
    }

    fn stage_output_dim(
        &self,
        stage: &Stage,
        buffer: BufferId,
        index: crate::ir::common::Index,
    ) -> Result<DimRef<BufferId>, LowerError> {
        stage
            .output
            .shape
            .0
            .iter()
            .position(|candidate| *candidate == index)
            .map(|dim| DimRef { buffer, dim })
            .ok_or_else(|| {
                LowerError::new(format!("index {} is not present in output shape", index.0))
            })
    }

    fn stage_axis(&self, stage: &Stage, axis_ref: StageAxisRef) -> Result<LiveAxis, LowerError> {
        match axis_ref {
            StageAxisRef::Local(axis) => match stage.axes.get(axis.0) {
                Some(Axis::Live { index, kind }) => Ok(LiveAxis {
                    index: *index,
                    kind: kind.clone(),
                }),
                Some(Axis::Pruned { .. }) => {
                    Err(LowerError::new(format!("axis {} is pruned", axis.0)))
                }
                None => Err(LowerError::new(format!("axis {} does not exist", axis.0))),
            },
            StageAxisRef::Consumer(axis) => Err(LowerError::new(format!(
                "consumer axis {} cannot define buffer layout",
                axis.0
            ))),
        }
    }

    fn lower_outputs(&self) -> Result<Vec<BufferId>, LowerError> {
        self.program
            .graph
            .outputs
            .iter()
            .copied()
            .map(|source| self.source_buffer(source))
            .collect()
    }

    fn lower_kernel(&mut self, stage_index: usize) -> Result<(), LowerError> {
        let mut kernel = KernelBuilder::new(self, stage_index);
        let body = kernel.emit_root_stage(stage_index)?;
        let (reads, writes) = kernel.params();
        let inputs = reads
            .iter()
            .copied()
            .map(|buffer| {
                self.buffer_sources[buffer.0].ok_or_else(|| {
                    LowerError::new(format!(
                        "kernel read buffer {} has no graph source",
                        buffer.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_id = NodeId(self.kernel_nodes.len());
        let outputs = writes
            .iter()
            .enumerate()
            .map(|(output, buffer)| {
                self.buffer_sources[buffer.0] = Some(Source::Node(kernel_id, OutputId(output)));
                GraphOutput
            })
            .collect();
        self.kernel_nodes.push(GraphNode {
            inner: Kernel {
                reads,
                writes,
                body,
            },
            inputs,
            outputs,
        });
        Ok(())
    }

    fn source_buffer(&self, source: Source) -> Result<BufferId, LowerError> {
        match source {
            Source::Input(input) => self.input_buffers.get(input.0).copied().ok_or_else(|| {
                LowerError::new(format!("source references nonexistent input {}", input.0))
            }),
            Source::Node(node, OutputId(0)) => {
                self.stage_buffers.get(node.0).copied().ok_or_else(|| {
                    LowerError::new(format!("source references nonexistent stage {}", node.0))
                })
            }
            Source::Node(_, output) => Err(LowerError::new(format!(
                "source references output {}",
                output.0
            ))),
        }
    }

    fn is_consumer_scoped_stage(&self, stage_index: usize) -> bool {
        self.program.graph.nodes[stage_index]
            .inner
            .axes
            .iter()
            .any(|axis| matches!(axis, Axis::Pruned { .. }))
    }
}

struct KernelBuilder<'a, 'b> {
    builder: &'b Builder<'a>,
    root: usize,
    next_loop: usize,
    reads: Vec<BufferId>,
    writes: Vec<BufferId>,
    accessed: Vec<BufferId>,
    written: Vec<BufferId>,
}

impl<'a, 'b> KernelBuilder<'a, 'b> {
    fn new(builder: &'b Builder<'a>, root: usize) -> Self {
        Self {
            builder,
            root,
            next_loop: 0,
            reads: Vec::new(),
            writes: Vec::new(),
            accessed: Vec::new(),
            written: Vec::new(),
        }
    }

    fn emit_root_stage(&mut self, stage_index: usize) -> Result<Block, LowerError> {
        if stage_index != self.root {
            return Err(LowerError::new("root stage mismatch"));
        }
        self.emit_stage(stage_index, &BTreeMap::new())
    }

    fn params(mut self) -> (Vec<BufferId>, Vec<BufferId>) {
        let written = self.written.iter().copied().collect::<BTreeSet<_>>();
        for buffer in self.accessed {
            if !written.contains(&buffer) && !self.reads.contains(&buffer) {
                self.reads.push(buffer);
            }
        }
        (self.reads, self.writes)
    }

    fn emit_stage(
        &mut self,
        stage_index: usize,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Block, LowerError> {
        let graph_node = &self.builder.program.graph.nodes[stage_index];
        let stage = &graph_node.inner;
        let output_buffer = self.builder.stage_buffers[stage_index];
        self.record_write(output_buffer);

        let mut local_loops = Vec::new();
        let mut stage_axes = BTreeMap::new();
        for (axis_index, axis) in stage.axes.iter().enumerate() {
            let axis_id = AxisId(axis_index);
            match axis {
                Axis::Live { index, kind } => {
                    let loop_id = self.alloc_loop();
                    let info = LoopInfo {
                        id: loop_id,
                        index: *index,
                        kind: kind.clone(),
                    };
                    local_loops.push((axis_id, info.clone()));
                    stage_axes.insert(axis_id, info);
                }
                Axis::Pruned { by } => {
                    let info = resolve_parent_axis(parent_axes, *by)?;
                    stage_axes.insert(axis_id, info);
                }
            }
        }

        let mut actions = vec![Vec::new(); local_loops.len() + 1];
        if let Some(site) = stage.output.init {
            let depth = self.site_depth(stage, &local_loops, site)?;
            actions[depth].push(Action::Init {
                op: stage.op,
                write: self.lower_access(output_buffer, &stage.output.layout, &stage_axes)?,
            });
        }

        for (input_index, input) in stage.inputs.iter().enumerate() {
            if let Some(site) = input.compute {
                let depth = self.site_depth(stage, &local_loops, site)?;
                let source = graph_node.inputs[input_index];
                let Source::Node(producer, OutputId(0)) = source else {
                    return Err(LowerError::new(format!(
                        "stage {} input {} has compute site without stage producer",
                        stage_index, input_index
                    )));
                };
                let fused = self.emit_stage(producer.0, &stage_axes)?;
                actions[depth].extend(fused.0);
            }
        }

        let reads = graph_node
            .inputs
            .iter()
            .copied()
            .zip(stage.inputs.iter())
            .map(|(source, input)| {
                let buffer = self.builder.source_buffer(source)?;
                self.lower_access(buffer, &input.layout, &stage_axes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let compute = Action::Compute {
            op: stage.op,
            write: self.lower_access(output_buffer, &stage.output.layout, &stage_axes)?,
            reads,
        };

        self.build_loop_block(stage_index, stage, local_loops.as_slice(), actions, compute)
    }

    fn build_loop_block(
        &mut self,
        stage_index: usize,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        actions: Vec<Vec<Action>>,
        compute: Action,
    ) -> Result<Block, LowerError> {
        fn build(
            emitter: &mut KernelBuilder<'_, '_>,
            stage: &Stage,
            stage_index: usize,
            local_loops: &[(AxisId, LoopInfo)],
            actions: &[Vec<Action>],
            compute: &Action,
            depth: usize,
        ) -> Result<Block, LowerError> {
            let mut block = actions[depth].clone();
            if depth == local_loops.len() {
                block.push(compute.clone());
            } else {
                let (axis, info) = local_loops[depth].clone();
                block.push(Action::Loop {
                    id: info.id,
                    extent: emitter.lower_loop_extent(stage_index, stage, axis, info.clone())?,
                    guard: TailGuard(emitter.tail_guard(stage, info)),
                    body: build(
                        emitter,
                        stage,
                        stage_index,
                        local_loops,
                        actions,
                        compute,
                        depth + 1,
                    )?,
                });
            }
            Ok(Block(block))
        }

        build(
            self,
            stage,
            stage_index,
            local_loops,
            actions.as_slice(),
            &compute,
            0,
        )
    }

    fn lower_loop_extent(
        &mut self,
        stage_index: usize,
        stage: &Stage,
        axis: AxisId,
        info: LoopInfo,
    ) -> Result<Extent<BufferId>, LowerError> {
        let source = self.semantic_dim_for_axis(stage_index, stage, info.index)?;
        self.record_access(source.buffer);
        let Axis::Live { kind, .. } = &stage.axes[axis.0] else {
            return Err(LowerError::new(format!("axis {} is not live", axis.0)));
        };
        Ok(Extent {
            source,
            kind: kind.clone(),
        })
    }

    fn semantic_dim_for_axis(
        &self,
        stage_index: usize,
        stage: &Stage,
        index: crate::ir::common::Index,
    ) -> Result<DimRef<BufferId>, LowerError> {
        if let Some(dim) = stage
            .output
            .shape
            .0
            .iter()
            .position(|candidate| *candidate == index)
        {
            return Ok(DimRef {
                buffer: self.builder.stage_buffers[stage_index],
                dim,
            });
        }

        for (input_index, input) in stage.inputs.iter().enumerate() {
            if let Some(dim) = input
                .shape
                .0
                .iter()
                .position(|candidate| *candidate == index)
            {
                let source = self.builder.program.graph.nodes[stage_index].inputs[input_index];
                return Ok(DimRef {
                    buffer: self.builder.source_buffer(source)?,
                    dim,
                });
            }
        }

        Err(LowerError::new(format!(
            "index {} has no semantic extent source",
            index.0
        )))
    }

    fn tail_guard(&self, stage: &Stage, info: LoopInfo) -> bool {
        if matches!(info.kind, ExtentKind::Semantic) {
            return false;
        }
        let level = extent_level(&info.kind);
        !stage.axes.iter().any(|candidate| match candidate {
            Axis::Live { index, kind } => *index == info.index && extent_level(kind) > level,
            Axis::Pruned { .. } => false,
        })
    }

    fn lower_access(
        &mut self,
        buffer: BufferId,
        layout: &Layout,
        axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Access, LowerError> {
        self.record_access(buffer);
        Ok(Access {
            buffer,
            index: layout
                .0
                .iter()
                .map(|dim| self.lower_iter(dim, axes))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn lower_iter(
        &self,
        dim: &LayoutDim,
        axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Iter, LowerError> {
        match dim {
            LayoutDim::Physical(axis) => Ok(Iter::Raw(resolve_stage_axis(axes, *axis)?.id)),
            LayoutDim::Semantic { axes: dim_axes, .. } => {
                let infos = dim_axes
                    .iter()
                    .map(|axis| resolve_stage_axis(axes, *axis))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Iter::Reconstructed {
                    loops: infos.iter().map(|info| info.id).collect(),
                    factors: reconstruction_factors(infos.as_slice()),
                })
            }
        }
    }

    fn site_depth(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        site: Site,
    ) -> Result<usize, LowerError> {
        match site {
            Site::Root => Ok(0),
            Site::At(axis) => match stage.axes.get(axis.0) {
                Some(Axis::Live { .. }) => local_loops
                    .iter()
                    .position(|(candidate, _)| *candidate == axis)
                    .map(|position| position + 1)
                    .ok_or_else(|| {
                        LowerError::new(format!("site references nonlocal axis {}", axis.0))
                    }),
                Some(Axis::Pruned { .. }) => Ok(0),
                None => Err(LowerError::new(format!(
                    "site references nonexistent axis {}",
                    axis.0
                ))),
            },
        }
    }

    fn alloc_loop(&mut self) -> LoopId {
        let loop_id = LoopId(self.next_loop);
        self.next_loop += 1;
        loop_id
    }

    fn record_access(&mut self, buffer: BufferId) {
        if !self.accessed.contains(&buffer) {
            self.accessed.push(buffer);
        }
    }

    fn record_write(&mut self, buffer: BufferId) {
        if !self.written.contains(&buffer) {
            self.written.push(buffer);
        }
        if !self.writes.contains(&buffer) {
            self.writes.push(buffer);
        }
        self.record_access(buffer);
    }
}

fn resolve_parent_axis(
    parent_axes: &BTreeMap<AxisId, LoopInfo>,
    axis: StageAxisRef,
) -> Result<LoopInfo, LowerError> {
    match axis {
        StageAxisRef::Local(axis) | StageAxisRef::Consumer(axis) => parent_axes
            .get(&axis)
            .cloned()
            .ok_or_else(|| LowerError::new(format!("consumer axis {} is not available", axis.0))),
    }
}

fn resolve_stage_axis(
    axes: &BTreeMap<AxisId, LoopInfo>,
    axis: StageAxisRef,
) -> Result<LoopInfo, LowerError> {
    match axis {
        StageAxisRef::Local(axis) | StageAxisRef::Consumer(axis) => axes
            .get(&axis)
            .cloned()
            .ok_or_else(|| LowerError::new(format!("axis {} is not available", axis.0))),
    }
}

fn reconstruction_factors(infos: &[LoopInfo]) -> Vec<usize> {
    infos
        .iter()
        .find_map(|info| match &info.kind {
            ExtentKind::Base(factors) => Some(factors.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            infos
                .iter()
                .filter_map(|info| match info.kind {
                    ExtentKind::Split { factor, .. } => Some(factor),
                    _ => None,
                })
                .collect()
        })
}

fn extent_level(kind: &ExtentKind) -> usize {
    match kind {
        ExtentKind::Semantic | ExtentKind::Base(_) => 0,
        ExtentKind::Split { level, .. } => *level,
    }
}

#[derive(Clone)]
struct LoopInfo {
    id: LoopId,
    index: crate::ir::common::Index,
    kind: ExtentKind,
}

#[derive(Clone)]
struct LiveAxis {
    index: crate::ir::common::Index,
    kind: ExtentKind,
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

    fn from_stage_graph(error: crate::check::stage::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_kernel_program(error: crate::check::kernel_program::ValidationError) -> Self {
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
    use super::lower_stage_program_to_kernel_program;
    use crate::front::parse_expr;
    use crate::ir::common::{DimRef, Extent, ExtentKind};
    use crate::ir::graph::{NodeId, OutputId, Source};
    use crate::ir::kernel_program::{Action, BufferKind, BufferLayout, Iter};
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::{component, front};

    fn lower_expr(src: &str) -> crate::ir::stage::StageProgram {
        lower_node_graph_to_stage_program(
            &lower_component_to_graph(&component::expr(parse_expr(src).unwrap())).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn lowers_single_stage_to_one_kernel() {
        let program = lower_stage_program_to_kernel_program(&lower_expr("ik*kj~ijk")).unwrap();

        assert_eq!(program.buffers.len(), 3);
        assert_eq!(program.buffers[0].kind, BufferKind::Input);
        assert_eq!(program.buffers[1].kind, BufferKind::Input);
        assert_eq!(program.buffers[2].kind, BufferKind::Output);
        assert_eq!(
            program.outputs,
            vec![crate::ir::kernel_program::BufferId(2)]
        );
        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(
            program.graph.nodes[0].inner.reads,
            vec![
                crate::ir::kernel_program::BufferId(0),
                crate::ir::kernel_program::BufferId(1)
            ]
        );
        assert_eq!(
            program.graph.nodes[0].inner.writes,
            vec![crate::ir::kernel_program::BufferId(2)]
        );
    }

    #[test]
    fn lowers_output_layout_to_buffer_layout() {
        let program =
            lower_stage_program_to_kernel_program(&lower_expr("ik*kj~ijk|i:2,k:8|iki'k'j"))
                .unwrap();

        assert_eq!(
            program.buffers[2].layout,
            BufferLayout(vec![
                Extent {
                    source: DimRef {
                        buffer: crate::ir::kernel_program::BufferId(2),
                        dim: 0
                    },
                    kind: ExtentKind::Semantic
                },
                Extent {
                    source: DimRef {
                        buffer: crate::ir::kernel_program::BufferId(2),
                        dim: 1
                    },
                    kind: ExtentKind::Semantic
                },
                Extent {
                    source: DimRef {
                        buffer: crate::ir::kernel_program::BufferId(2),
                        dim: 2
                    },
                    kind: ExtentKind::Semantic
                },
            ])
        );
    }

    #[test]
    fn lowers_non_compute_chain_to_two_kernels() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        assert_eq!(program.graph.nodes.len(), 2);
        assert_eq!(
            program.graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
    }

    #[test]
    fn passes_through_program_input_output_without_kernel() {
        let stage_program = crate::ir::stage::StageProgram {
            shapes: crate::ir::stage::ShapeTable {
                inputs: vec![crate::ir::stage::ProgramShape(vec![
                    crate::ir::stage::ShapeDim {
                        input: crate::ir::graph::InputId(0),
                        dim: 0,
                    },
                ])],
                nodes: vec![],
            },
            graph: crate::ir::graph::Graph {
                inputs: vec![crate::ir::graph::Input],
                nodes: vec![],
                outputs: vec![Source::Input(crate::ir::graph::InputId(0))],
            },
        };

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        assert_eq!(program.buffers.len(), 1);
        assert_eq!(
            program.outputs,
            vec![crate::ir::kernel_program::BufferId(0)]
        );
        assert!(program.graph.nodes.is_empty());
        assert_eq!(
            program.graph.outputs,
            vec![Source::Input(crate::ir::graph::InputId(0))]
        );
    }

    #[test]
    fn keeps_materialized_producer_and_compute_at_duplicate() {
        let producer = component::expr(front::parse_expr("ik*kj~ijk").unwrap());
        let compute_consumer = component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap());
        let materialized_consumer = component::expr(front::parse_expr("+ijk~ij").unwrap());
        let component = producer.chain(compute_consumer.fanout(materialized_consumer));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let mut ops = Vec::new();
        for node in &program.graph.nodes {
            collect_compute_ops(&node.inner.body, &mut ops);
        }

        assert_eq!(program.graph.nodes.len(), 3);
        assert_eq!(
            ops.iter()
                .filter(|op| **op == crate::ir::common::Op::Mul)
                .count(),
            2
        );
        assert_eq!(
            ops.iter()
                .filter(|op| **op == crate::ir::common::Op::Add)
                .count(),
            2
        );
        assert!(program
            .graph
            .nodes
            .iter()
            .any(|node| node.inner.writes.len() == 2));
    }

    #[test]
    fn absorbs_compute_at_producer_into_consumer_kernel() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(
            program.graph.nodes[0].inner.reads,
            vec![
                crate::ir::kernel_program::BufferId(0),
                crate::ir::kernel_program::BufferId(1)
            ]
        );
        assert_eq!(
            program.graph.nodes[0].inner.writes,
            vec![
                crate::ir::kernel_program::BufferId(3),
                crate::ir::kernel_program::BufferId(2),
            ]
        );
    }

    #[test]
    fn lowers_fully_staged_producer_to_scalar_layout() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ijk0").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        assert_eq!(program.buffers[2].layout, BufferLayout(vec![]));
    }

    #[test]
    fn lowers_partially_staged_producer_to_remaining_axis_layout() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        assert_eq!(
            program.buffers[2].layout,
            BufferLayout(vec![Extent {
                source: DimRef {
                    buffer: crate::ir::kernel_program::BufferId(2),
                    dim: 2,
                },
                kind: ExtentKind::Semantic,
            }])
        );
    }

    #[test]
    fn emits_fused_compute_before_consumer_compute() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let mut computes = Vec::new();
        collect_compute_writes(&program.graph.nodes[0].inner.body, &mut computes);
        assert_eq!(
            computes,
            vec![
                crate::ir::kernel_program::BufferId(2),
                crate::ir::kernel_program::BufferId(3),
            ]
        );
    }

    #[test]
    fn emits_reduction_init_before_fused_reduction_loop() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let body = first_loop_body(&program.graph.nodes[0].inner.body);
        let body = first_loop_body(body);
        assert!(matches!(body.0[0], Action::Init { .. }));
        assert!(matches!(body.0[1], Action::Loop { .. }));
    }

    #[test]
    fn emits_tail_guard_on_innermost_split_loop() {
        let program = lower_stage_program_to_kernel_program(&lower_expr("ik~ik|i:8|ii'k")).unwrap();
        let mut guards = Vec::new();
        collect_loop_guards(&program.graph.nodes[0].inner.body, &mut guards);

        assert_eq!(guards, vec![false, true, false]);
    }

    #[test]
    fn reconstructs_semantic_accesses() {
        let program = lower_stage_program_to_kernel_program(&lower_expr("ik~ik|i:8|ii'k")).unwrap();
        let compute = first_compute(&program.graph.nodes[0].inner.body).unwrap();
        let Action::Compute { reads, .. } = compute else {
            unreachable!();
        };
        assert!(matches!(reads[0].index[0], Iter::Reconstructed { .. }));
    }

    #[test]
    fn preserves_multi_factor_reconstruction_factors() {
        let program =
            lower_stage_program_to_kernel_program(&lower_expr("ik~ik|i:2:4|ii'i''k")).unwrap();
        let compute = first_compute(&program.graph.nodes[0].inner.body).unwrap();
        let Action::Compute { reads, .. } = compute else {
            unreachable!();
        };
        let Iter::Reconstructed { loops, factors } = &reads[0].index[0] else {
            unreachable!();
        };

        assert_eq!(loops.len(), 3);
        assert_eq!(factors, &vec![2, 4]);
    }

    fn collect_compute_writes(
        block: &crate::ir::kernel_program::Block,
        writes: &mut Vec<crate::ir::kernel_program::BufferId>,
    ) {
        for action in &block.0 {
            match action {
                Action::Loop { body, .. } => collect_compute_writes(body, writes),
                Action::Compute { write, .. } => writes.push(write.buffer),
                Action::Init { .. } => {}
            }
        }
    }

    fn collect_compute_ops(
        block: &crate::ir::kernel_program::Block,
        ops: &mut Vec<crate::ir::common::Op>,
    ) {
        for action in &block.0 {
            match action {
                Action::Loop { body, .. } => collect_compute_ops(body, ops),
                Action::Compute { op, .. } => ops.push(*op),
                Action::Init { .. } => {}
            }
        }
    }

    fn collect_loop_guards(block: &crate::ir::kernel_program::Block, guards: &mut Vec<bool>) {
        for action in &block.0 {
            if let Action::Loop { guard, body, .. } = action {
                guards.push(guard.0);
                collect_loop_guards(body, guards);
            }
        }
    }

    fn first_compute(
        block: &crate::ir::kernel_program::Block,
    ) -> Option<&crate::ir::kernel_program::Action> {
        for action in &block.0 {
            match action {
                Action::Compute { .. } => return Some(action),
                Action::Loop { body, .. } => {
                    if let Some(action) = first_compute(body) {
                        return Some(action);
                    }
                }
                Action::Init { .. } => {}
            }
        }
        None
    }

    fn first_loop_body(
        block: &crate::ir::kernel_program::Block,
    ) -> &crate::ir::kernel_program::Block {
        for action in &block.0 {
            if let Action::Loop { body, .. } = action {
                return body;
            }
        }
        panic!("expected loop action");
    }
}

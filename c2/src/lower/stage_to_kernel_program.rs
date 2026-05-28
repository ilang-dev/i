use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::kernel_program::validate_kernel_program;
use crate::check::stage::validate_stage_graph;
use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
use crate::ir::graph::{
    Graph, InputId, Node as GraphNode, NodeId, Output as GraphOutput, OutputId, Source,
};
use crate::ir::kernel_program::{
    Access, Action, Block, Buffer, BufferId, BufferKind, BufferLayout, BufferShape, Iter, Kernel,
    KernelProgram, LoopId, ScalarExpr, TailGuard,
};
use crate::ir::stage::{
    Axis, AxisId, AxisRef as StageAxisRef, Layout, LayoutDim, ProgramShape, Shape, ShapeDim, Site,
    Stage, StageProgram,
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
            if self.is_fused_only_stage(stage_index) {
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

    fn add_intermediate_like(&mut self, source: BufferId) -> Result<BufferId, LowerError> {
        let buffer = self.buffers.get(source.0).cloned().ok_or_else(|| {
            LowerError::new(format!(
                "snapshot source buffer {} does not exist",
                source.0
            ))
        })?;
        let id = BufferId(self.buffers.len());
        self.buffer_sources.push(None);
        self.buffers.push(Buffer {
            kind: BufferKind::Intermediate,
            shape: buffer.shape,
            layout: buffer.layout,
        });
        Ok(id)
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

    fn is_fused_only_stage(&self, stage_index: usize) -> bool {
        let stage_id = NodeId(stage_index);
        self.program.graph.nodes.iter().any(|node| {
            node.inputs
                .iter()
                .copied()
                .zip(node.inner.inputs.iter())
                .any(|(source, input)| {
                    input.compute.is_some() && source == Source::Node(stage_id, OutputId(0))
                })
        })
    }
}

struct KernelBuilder<'a, 'b> {
    builder: &'b mut Builder<'a>,
    root: usize,
    next_loop: usize,
    reads: Vec<BufferId>,
    writes: Vec<BufferId>,
    accessed: Vec<BufferId>,
    written: Vec<BufferId>,
    stage_aliases: BTreeMap<usize, usize>,
    emitted_stages: Vec<EmittedStageKey>,
    applied_corrections: BTreeSet<(usize, BufferId, BufferId, usize)>,
}

impl<'a, 'b> KernelBuilder<'a, 'b> {
    fn new(builder: &'b mut Builder<'a>, root: usize) -> Self {
        Self {
            builder,
            root,
            next_loop: 0,
            reads: Vec::new(),
            writes: Vec::new(),
            accessed: Vec::new(),
            written: Vec::new(),
            stage_aliases: BTreeMap::new(),
            emitted_stages: Vec::new(),
            applied_corrections: BTreeSet::new(),
        }
    }

    fn emit_root_stage(&mut self, stage_index: usize) -> Result<Block, LowerError> {
        if stage_index != self.root {
            return Err(LowerError::new("root stage mismatch"));
        }
        let emitted = self.emit_stage(stage_index, &BTreeMap::new())?;
        if let Some(hoisted) = emitted.hoisted.first() {
            return Err(LowerError::new(format!(
                "stage {} left fused actions hoisted to unavailable {}",
                stage_index,
                format_hoist_site(hoisted.site)
            )));
        }
        if !emitted.corrections.is_empty() {
            return Err(LowerError::new(
                "online reduction correction reached root without an accumulator",
            ));
        }
        Ok(emitted.block)
    }

    fn params(mut self) -> (Vec<BufferId>, Vec<BufferId>) {
        let writes = self
            .writes
            .iter()
            .copied()
            .map(|buffer| self.canonical_buffer(buffer))
            .collect::<Vec<_>>();
        let written = self
            .written
            .iter()
            .copied()
            .map(|buffer| self.canonical_buffer(buffer))
            .collect::<BTreeSet<_>>();
        let accessed = self
            .accessed
            .iter()
            .copied()
            .map(|buffer| self.canonical_buffer(buffer))
            .collect::<Vec<_>>();
        self.reads = self
            .reads
            .iter()
            .copied()
            .map(|buffer| self.canonical_buffer(buffer))
            .collect();
        for buffer in accessed {
            if !written.contains(&buffer) && !self.reads.contains(&buffer) {
                self.reads.push(buffer);
            }
        }
        (self.reads, writes)
    }

    fn emit_stage(
        &mut self,
        stage_index: usize,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<EmittedStage, LowerError> {
        let graph_node = &self.builder.program.graph.nodes[stage_index];
        let stage = &graph_node.inner;
        let output_buffer = self.builder.stage_buffers[stage_index];

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
                Axis::Pruned { by, .. } => {
                    let info = resolve_parent_axis(parent_axes, *by)?;
                    stage_axes.insert(axis_id, info);
                }
            }
        }

        let mut actions = vec![Vec::new(); local_loops.len() + 1];
        let mut hoisted = Vec::new();
        let online_correction =
            self.online_correction(stage_index, stage, output_buffer, &stage_axes)?;
        if let Some(site) = stage.output.init {
            match self.init_placement(stage, &local_loops, site)? {
                Placement::Local(depth) => {
                    self.validate_init_depth(stage, &local_loops, depth)?;
                    actions[depth].push(Action::Init {
                        op: stage.op,
                        write: self.lower_access(
                            output_buffer,
                            &stage.output.layout,
                            &stage_axes,
                            parent_axes,
                        )?,
                        zero_checks: self.init_zero_checks(
                            stage,
                            &local_loops,
                            &stage_axes,
                            depth,
                        )?,
                    });
                }
                Placement::Hoist(site) => {
                    hoisted.push(HoistedBlock {
                        site,
                        block: Block(vec![Action::Init {
                            op: stage.op,
                            write: self.lower_access(
                                output_buffer,
                                &stage.output.layout,
                                &stage_axes,
                                parent_axes,
                            )?,
                            zero_checks: Vec::new(),
                        }]),
                    });
                }
            }
        }
        if let Some(correction) = &online_correction {
            let layout = self.semantic_shape_layout(&correction.shape, stage)?;
            let depth = self.layout_depth(&layout, &local_loops)?;
            actions[depth].push(self.snapshot_action(
                correction,
                stage,
                &stage_axes,
                parent_axes,
            )?);
        }

        let mut hoisted_corrections = Vec::new();
        let mut corrections = Vec::new();
        for (input_index, input) in stage.inputs.iter().enumerate() {
            if let Some(site) = input.compute {
                let source = graph_node.inputs[input_index];
                let Source::Node(producer, OutputId(0)) = source else {
                    return Err(LowerError::new(format!(
                        "stage {} input {} has compute site without stage producer",
                        stage_index, input_index
                    )));
                };
                let fused = self.emit_stage(producer.0, &stage_axes)?;
                self.place_hoisted_blocks(
                    stage,
                    &local_loops,
                    &mut actions,
                    &mut hoisted,
                    fused.hoisted,
                )?;
                self.place_hoisted_corrections(
                    stage,
                    stage_index,
                    output_buffer,
                    &stage_axes,
                    &local_loops,
                    &mut actions,
                    &mut hoisted_corrections,
                    &mut corrections,
                    fused.hoisted_corrections,
                    parent_axes,
                )?;
                match self.compute_placement(stage, &local_loops, site)? {
                    Placement::Local(depth) => {
                        actions[depth].extend(fused.block.0);
                        self.handle_online_corrections_at_stage(
                            stage,
                            stage_index,
                            input_index,
                            output_buffer,
                            &stage_axes,
                            &local_loops,
                            &mut actions,
                            &mut corrections,
                            fused.corrections,
                            Some(depth),
                            parent_axes,
                        )?;
                    }
                    Placement::Hoist(site) => {
                        if self.can_apply_online_corrections(stage, fused.corrections.as_slice()) {
                            self.handle_online_corrections_at_stage(
                                stage,
                                stage_index,
                                input_index,
                                output_buffer,
                                &stage_axes,
                                &local_loops,
                                &mut actions,
                                &mut corrections,
                                fused.corrections,
                                None,
                                parent_axes,
                            )?;
                            hoisted.push(HoistedBlock {
                                site,
                                block: fused.block,
                            });
                        } else {
                            hoisted.push(HoistedBlock {
                                site,
                                block: fused.block,
                            });
                            hoisted_corrections.push(HoistedCorrections {
                                site,
                                corrections: self.propagate_online_corrections(
                                    stage,
                                    input_index,
                                    fused.corrections,
                                )?,
                            });
                        }
                    }
                }
            } else if let Source::Node(producer, OutputId(0)) = graph_node.inputs[input_index] {
                let incoming = self.emitted_corrections(producer);
                self.handle_online_corrections_at_stage(
                    stage,
                    stage_index,
                    input_index,
                    output_buffer,
                    &stage_axes,
                    &local_loops,
                    &mut actions,
                    &mut corrections,
                    incoming,
                    None,
                    parent_axes,
                )?;
            }
        }
        if let Some(correction) = online_correction {
            corrections.push(correction);
        }

        if stage_index != self.root {
            let key = self.stage_key(stage_index, stage, &graph_node.inputs);
            if let Some(existing) = self
                .emitted_stages
                .iter()
                .find(|existing| existing.stage == *stage && existing.inputs == key.inputs)
            {
                if existing.stage_index != stage_index
                    && stage
                        .axes
                        .iter()
                        .any(|axis| matches!(axis, Axis::Pruned { .. }))
                    && !stage.output.layout.0.is_empty()
                {
                    return Err(LowerError::new(format!(
                        "shared non-root compute-at placement for stage {} is not supported",
                        stage_index
                    )));
                }
                self.stage_aliases.insert(stage_index, existing.stage_index);
                return Ok(EmittedStage {
                    block: Block(Vec::new()),
                    hoisted: existing
                        .hoisted
                        .iter()
                        .map(|hoisted| HoistedBlock {
                            site: hoisted.site,
                            block: Block(Vec::new()),
                        })
                        .collect(),
                    hoisted_corrections: existing.hoisted_corrections.clone(),
                    corrections: existing.corrections.clone(),
                });
            }
            let mut key = key;
            key.hoisted = hoisted.clone();
            key.hoisted_corrections = hoisted_corrections.clone();
            key.corrections = corrections.clone();
            self.emitted_stages.push(key);
        }

        self.record_write(output_buffer);

        let reads = graph_node
            .inputs
            .iter()
            .copied()
            .zip(stage.inputs.iter())
            .map(|(source, input)| self.lower_read(source, input, stage, &stage_axes, parent_axes))
            .collect::<Result<Vec<_>, _>>()?;
        let compute = Action::Compute {
            op: stage.op,
            write: self.lower_access(
                output_buffer,
                &stage.output.layout,
                &stage_axes,
                parent_axes,
            )?,
            reads,
        };

        Ok(EmittedStage {
            block: self.build_loop_block(
                stage_index,
                stage,
                local_loops.as_slice(),
                actions,
                compute,
            )?,
            hoisted,
            hoisted_corrections,
            corrections,
        })
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
        for (input_index, input) in stage.inputs.iter().enumerate() {
            if let Some(dim) = input
                .shape
                .0
                .iter()
                .position(|candidate| *candidate == index)
            {
                let source = self.builder.program.graph.nodes[stage_index].inputs[input_index];
                return Ok(DimRef {
                    buffer: self.source_buffer(source)?,
                    dim,
                });
            }
        }

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
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Access, LowerError> {
        self.record_access(buffer);
        Ok(Access {
            buffer,
            index: layout
                .0
                .iter()
                .map(|dim| self.lower_iter(dim, axes, parent_axes))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn source_buffer(&self, source: Source) -> Result<BufferId, LowerError> {
        match source {
            Source::Node(node, output) => {
                let canonical = self.canonical_stage(node);
                self.builder.source_buffer(Source::Node(canonical, output))
            }
            Source::Input(_) => self.builder.source_buffer(source),
        }
    }

    fn canonical_stage(&self, node: NodeId) -> NodeId {
        let mut index = node.0;
        while let Some(next) = self.stage_aliases.get(&index).copied() {
            if next == index {
                break;
            }
            index = next;
        }
        NodeId(index)
    }

    fn canonical_buffer(&self, buffer: BufferId) -> BufferId {
        let Some(stage_index) = self
            .builder
            .stage_buffers
            .iter()
            .position(|candidate| *candidate == buffer)
        else {
            return buffer;
        };
        self.builder.stage_buffers[self.canonical_stage(NodeId(stage_index)).0]
    }

    fn stage_key(&self, stage_index: usize, stage: &Stage, inputs: &[Source]) -> EmittedStageKey {
        EmittedStageKey {
            stage_index,
            stage: stage.clone(),
            inputs: inputs
                .iter()
                .copied()
                .map(|source| match source {
                    Source::Node(node, output) => Source::Node(self.canonical_stage(node), output),
                    Source::Input(_) => source,
                })
                .collect(),
            hoisted: Vec::new(),
            hoisted_corrections: Vec::new(),
            corrections: Vec::new(),
        }
    }

    fn lower_iter(
        &self,
        dim: &LayoutDim,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Iter, LowerError> {
        match dim {
            LayoutDim::Physical(axis) => {
                Ok(Iter::Raw(resolve_stage_axis(axes, parent_axes, *axis)?.id))
            }
            LayoutDim::Semantic { axes: dim_axes, .. } => {
                let infos = dim_axes
                    .iter()
                    .map(|axis| resolve_stage_axis(axes, parent_axes, *axis))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Iter::Reconstructed {
                    loops: infos.iter().map(|info| info.id).collect(),
                    factors: reconstruction_factors(infos.as_slice()),
                })
            }
        }
    }

    fn compute_placement(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        site: Site,
    ) -> Result<Placement, LowerError> {
        match site {
            Site::Root => self.root_placement(stage),
            Site::At(axis) => self.axis_placement(stage, local_loops, axis),
        }
    }

    fn init_placement(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        site: Site,
    ) -> Result<Placement, LowerError> {
        match site {
            Site::Root => self.root_placement(stage),
            Site::At(axis) => self.axis_placement(stage, local_loops, axis),
        }
    }

    fn root_placement(&self, stage: &Stage) -> Result<Placement, LowerError> {
        for axis in &stage.axes {
            match axis {
                Axis::Live { .. } => return Ok(Placement::Local(0)),
                Axis::Pruned {
                    by: StageAxisRef::Consumer(axis),
                    ..
                } => return Ok(Placement::Hoist(HoistSite::Before(*axis))),
                Axis::Pruned {
                    by: StageAxisRef::Local(_),
                    ..
                } => {}
            }
        }
        Ok(Placement::Local(0))
    }

    fn axis_placement(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        axis: AxisId,
    ) -> Result<Placement, LowerError> {
        match stage.axes.get(axis.0) {
            Some(Axis::Live { .. }) => local_loops
                .iter()
                .position(|(candidate, _)| *candidate == axis)
                .map(|position| Placement::Local(position + 1))
                .ok_or_else(|| {
                    LowerError::new(format!("site references nonlocal axis {}", axis.0))
                }),
            Some(Axis::Pruned {
                by: StageAxisRef::Consumer(axis),
                ..
            }) => Ok(Placement::Hoist(HoistSite::After(*axis))),
            Some(Axis::Pruned {
                by: StageAxisRef::Local(axis),
                ..
            }) => self.axis_placement(stage, local_loops, *axis),
            None => Err(LowerError::new(format!(
                "site references nonexistent axis {}",
                axis.0
            ))),
        }
    }

    fn place_hoisted_blocks(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        actions: &mut [Vec<Action>],
        propagated: &mut Vec<HoistedBlock>,
        hoisted: Vec<HoistedBlock>,
    ) -> Result<(), LowerError> {
        for hoist in hoisted {
            match self.hoist_placement(stage, local_loops, hoist.site)? {
                Placement::Local(depth) => actions[depth].extend(hoist.block.0),
                Placement::Hoist(site) => propagated.push(HoistedBlock {
                    site,
                    block: hoist.block,
                }),
            }
        }
        Ok(())
    }

    fn place_hoisted_corrections(
        &mut self,
        stage: &Stage,
        stage_index: usize,
        output_buffer: BufferId,
        stage_axes: &BTreeMap<AxisId, LoopInfo>,
        local_loops: &[(AxisId, LoopInfo)],
        actions: &mut [Vec<Action>],
        propagated: &mut Vec<HoistedCorrections>,
        corrections: &mut Vec<OnlineCorrection>,
        hoisted: Vec<HoistedCorrections>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<(), LowerError> {
        for hoist in hoisted {
            match self.hoist_placement(stage, local_loops, hoist.site)? {
                Placement::Local(depth) => {
                    if self.can_apply_online_corrections(stage, hoist.corrections.as_slice()) {
                        self.apply_online_corrections_at_stage(
                            stage,
                            stage_index,
                            output_buffer,
                            stage_axes,
                            local_loops,
                            actions,
                            corrections,
                            hoist.corrections,
                            Some(depth),
                            parent_axes,
                        )?;
                    } else {
                        let propagated_corrections =
                            self.propagate_online_corrections(stage, 0, hoist.corrections)?;
                        if !propagated_corrections.is_empty() {
                            return Err(LowerError::new(
                                "online correction was hoisted to a non-accumulator stage",
                            ));
                        }
                    }
                }
                Placement::Hoist(site) => {
                    if self.can_apply_online_corrections(stage, hoist.corrections.as_slice()) {
                        self.apply_online_corrections_at_stage(
                            stage,
                            stage_index,
                            output_buffer,
                            stage_axes,
                            local_loops,
                            actions,
                            corrections,
                            hoist.corrections,
                            None,
                            parent_axes,
                        )?;
                        let _ = site;
                    } else {
                        propagated.push(HoistedCorrections {
                            site,
                            corrections: self.propagate_online_corrections(
                                stage,
                                0,
                                hoist.corrections,
                            )?,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn hoist_placement(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        site: HoistSite,
    ) -> Result<Placement, LowerError> {
        match site {
            HoistSite::Before(axis) => self.axis_before_placement(stage, local_loops, axis),
            HoistSite::After(axis) => self.axis_placement(stage, local_loops, axis),
        }
    }

    fn axis_before_placement(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        axis: AxisId,
    ) -> Result<Placement, LowerError> {
        match stage.axes.get(axis.0) {
            Some(Axis::Live { .. }) => local_loops
                .iter()
                .position(|(candidate, _)| *candidate == axis)
                .map(Placement::Local)
                .ok_or_else(|| {
                    LowerError::new(format!("site references nonlocal axis {}", axis.0))
                }),
            Some(Axis::Pruned {
                by: StageAxisRef::Consumer(axis),
                ..
            }) => Ok(Placement::Hoist(HoistSite::Before(*axis))),
            Some(Axis::Pruned {
                by: StageAxisRef::Local(axis),
                ..
            }) => self.axis_before_placement(stage, local_loops, *axis),
            None => Err(LowerError::new(format!(
                "site references nonexistent axis {}",
                axis.0
            ))),
        }
    }

    fn validate_init_depth(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        depth: usize,
    ) -> Result<(), LowerError> {
        let output_indexes = stage
            .output
            .shape
            .0
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for (position, (_, info)) in local_loops.iter().enumerate() {
            if output_indexes.contains(&info.index) && position >= depth {
                return Err(LowerError::new(format!(
                    "init site is outside output loop {}",
                    info.index.0
                )));
            }
        }
        Ok(())
    }

    fn reject_unimplemented_online_reduction(
        &self,
        stage_index: usize,
        stage: &Stage,
    ) -> Result<(), LowerError> {
        if stage.output.init.is_none() {
            return Ok(());
        }

        let output_indexes = stage
            .output
            .shape
            .0
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let needs_online = stage.axes.iter().any(|axis| match axis {
            Axis::Pruned { index, .. } => !output_indexes.contains(index),
            Axis::Live { .. } => false,
        });

        if needs_online {
            return Err(LowerError::new(format!(
                "stage {} requires online reduction lowering: fused reduction axis is consumed before completion",
                stage_index
            )));
        }

        Ok(())
    }

    fn online_correction(
        &mut self,
        stage_index: usize,
        stage: &Stage,
        output_buffer: BufferId,
        _stage_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Option<OnlineCorrection>, LowerError> {
        if stage_index == self.root || !requires_online_reduction(stage) {
            return Ok(None);
        }
        let kind = match stage.op {
            Op::Add => OnlineCorrectionKind::Divisor,
            Op::Max => OnlineCorrectionKind::ExpShift,
            _ => {
                self.reject_unimplemented_online_reduction(stage_index, stage)?;
                return Ok(None);
            }
        };

        let old_buffer = self.builder.add_intermediate_like(output_buffer)?;
        self.record_write(old_buffer);
        Ok(Some(OnlineCorrection {
            old_buffer,
            new_buffer: output_buffer,
            shape: stage.output.shape.clone(),
            kind,
            state: OnlineCorrectionState::Pending,
        }))
    }

    fn can_apply_online_corrections(
        &self,
        stage: &Stage,
        corrections: &[OnlineCorrection],
    ) -> bool {
        stage.op == Op::Add
            && stage.output.init.is_some()
            && corrections.iter().all(|correction| {
                correction.state.is_ready()
                    && correction_shape_broadcasts_to_stage(correction, stage)
            })
    }

    fn snapshot_action(
        &mut self,
        correction: &OnlineCorrection,
        stage: &Stage,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Action, LowerError> {
        Ok(Action::Snapshot {
            write: self.lower_semantic_shape_access(
                correction.old_buffer,
                &correction.shape,
                stage,
                axes,
                parent_axes,
            )?,
            read: self.lower_semantic_shape_access(
                correction.new_buffer,
                &correction.shape,
                stage,
                axes,
                parent_axes,
            )?,
        })
    }

    fn scale_action(
        &mut self,
        correction: &OnlineCorrection,
        stage: &Stage,
        write: Access,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Action, LowerError> {
        let old = self.lower_semantic_shape_access(
            correction.old_buffer,
            &correction.shape,
            stage,
            axes,
            parent_axes,
        )?;
        let new = self.lower_semantic_shape_access(
            correction.new_buffer,
            &correction.shape,
            stage,
            axes,
            parent_axes,
        )?;
        let factor = match correction.kind {
            OnlineCorrectionKind::Divisor => ScalarExpr::Binary {
                op: Op::Div,
                lhs: Box::new(ScalarExpr::Access(old)),
                rhs: Box::new(ScalarExpr::Access(new)),
            },
            OnlineCorrectionKind::ExpShift => ScalarExpr::Unary {
                op: Op::Pow,
                arg: Box::new(ScalarExpr::Binary {
                    op: Op::Sub,
                    lhs: Box::new(ScalarExpr::Access(old)),
                    rhs: Box::new(ScalarExpr::Access(new)),
                }),
            },
        };
        Ok(Action::Scale { write, factor })
    }

    fn lower_read(
        &mut self,
        source: Source,
        input: &crate::ir::stage::Input,
        stage: &Stage,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Access, LowerError> {
        let buffer = self.source_buffer(source)?;
        if let Source::Node(node, OutputId(0)) = source {
            if input.compute.is_some() {
                let producer = &self.builder.program.graph.nodes[node.0].inner;
                let semantic_layout = self.semantic_shape_layout(&producer.output.shape, stage)?;
                if output_uses_semantic_layout(producer)
                    && self.layout_axes_are_available(&semantic_layout, axes, parent_axes)
                {
                    return self.lower_access(buffer, &semantic_layout, axes, parent_axes);
                }
            }
        }
        self.lower_access(buffer, &input.layout, axes, parent_axes)
    }

    fn lower_semantic_shape_access(
        &mut self,
        buffer: BufferId,
        shape: &Shape,
        stage: &Stage,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<Access, LowerError> {
        let layout = self.semantic_shape_layout(shape, stage)?;
        self.lower_access(buffer, &layout, axes, parent_axes)
    }

    fn semantic_shape_layout(&self, shape: &Shape, stage: &Stage) -> Result<Layout, LowerError> {
        shape
            .0
            .iter()
            .copied()
            .map(|index| {
                Ok(LayoutDim::Semantic {
                    index,
                    axes: stage_axis_refs_for_index(stage, index)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Layout)
    }

    fn accumulator_output_depth(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
    ) -> Result<usize, LowerError> {
        let output_indexes = stage
            .output
            .shape
            .0
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        Ok(local_loops
            .iter()
            .enumerate()
            .filter_map(|(position, (_, info))| {
                output_indexes.contains(&info.index).then_some(position + 1)
            })
            .max()
            .unwrap_or(0))
    }

    fn layout_axes_are_available(
        &self,
        layout: &Layout,
        axes: &BTreeMap<AxisId, LoopInfo>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> bool {
        layout.0.iter().all(|dim| match dim {
            LayoutDim::Physical(axis) => resolve_stage_axis(axes, parent_axes, *axis).is_ok(),
            LayoutDim::Semantic { axes: dim_axes, .. } => dim_axes
                .iter()
                .all(|axis| resolve_stage_axis(axes, parent_axes, *axis).is_ok()),
        })
    }

    fn layout_depth(
        &self,
        layout: &Layout,
        local_loops: &[(AxisId, LoopInfo)],
    ) -> Result<usize, LowerError> {
        let local_positions = local_loops
            .iter()
            .enumerate()
            .map(|(position, (axis, _))| (*axis, position))
            .collect::<BTreeMap<_, _>>();
        let mut depth = 0;

        for dim in &layout.0 {
            match dim {
                LayoutDim::Physical(axis) => {
                    depth = depth.max(local_axis_depth(*axis, &local_positions)?);
                }
                LayoutDim::Semantic { axes, .. } => {
                    for axis in axes {
                        depth = depth.max(local_axis_depth(*axis, &local_positions)?);
                    }
                }
            }
        }

        Ok(depth)
    }

    fn handle_online_corrections_at_stage(
        &mut self,
        stage: &Stage,
        stage_index: usize,
        input_index: usize,
        output_buffer: BufferId,
        stage_axes: &BTreeMap<AxisId, LoopInfo>,
        local_loops: &[(AxisId, LoopInfo)],
        actions: &mut [Vec<Action>],
        outgoing: &mut Vec<OnlineCorrection>,
        incoming: Vec<OnlineCorrection>,
        producer_depth: Option<usize>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<(), LowerError> {
        let incoming = dedup_online_corrections(incoming);
        if incoming.is_empty() {
            return Ok(());
        }
        if self.can_apply_online_corrections(stage, incoming.as_slice()) {
            self.apply_online_corrections_at_stage(
                stage,
                stage_index,
                output_buffer,
                stage_axes,
                local_loops,
                actions,
                outgoing,
                incoming,
                producer_depth,
                parent_axes,
            )?;
        } else {
            outgoing.extend(self.propagate_online_corrections(stage, input_index, incoming)?);
        }
        Ok(())
    }

    fn apply_online_corrections_at_stage(
        &mut self,
        stage: &Stage,
        stage_index: usize,
        output_buffer: BufferId,
        stage_axes: &BTreeMap<AxisId, LoopInfo>,
        local_loops: &[(AxisId, LoopInfo)],
        actions: &mut [Vec<Action>],
        outgoing: &mut Vec<OnlineCorrection>,
        incoming: Vec<OnlineCorrection>,
        producer_depth: Option<usize>,
        parent_axes: &BTreeMap<AxisId, LoopInfo>,
    ) -> Result<(), LowerError> {
        let incoming = dedup_online_corrections(incoming);
        if incoming.is_empty() {
            return Ok(());
        }
        self.validate_corrections_ready(incoming.as_slice())?;
        let write =
            self.lower_access(output_buffer, &stage.output.layout, stage_axes, parent_axes)?;
        let accumulator_depth = self.accumulator_output_depth(stage, local_loops)?;
        let scale_depth = producer_depth
            .map(|depth| depth.max(accumulator_depth))
            .unwrap_or(accumulator_depth);
        let scale_corrections = incoming
            .iter()
            .filter(|correction| {
                self.applied_corrections.insert((
                    stage_index,
                    correction.old_buffer,
                    correction.new_buffer,
                    correction_kind_key(correction.kind),
                ))
            })
            .collect::<Vec<_>>();
        let scales = scale_corrections
            .into_iter()
            .map(|correction| {
                self.scale_action(correction, stage, write.clone(), stage_axes, parent_axes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        actions[scale_depth].extend(scales);
        if stage_index != self.root {
            outgoing.extend(incoming);
        }
        Ok(())
    }

    fn emitted_corrections(&self, stage: NodeId) -> Vec<OnlineCorrection> {
        let canonical = self.canonical_stage(stage);
        self.emitted_stages
            .iter()
            .find(|emitted| emitted.stage_index == canonical.0)
            .map(|emitted| emitted.corrections.clone())
            .unwrap_or_default()
    }

    fn propagate_online_corrections(
        &self,
        stage: &Stage,
        input_index: usize,
        corrections: Vec<OnlineCorrection>,
    ) -> Result<Vec<OnlineCorrection>, LowerError> {
        if corrections.is_empty() {
            return Ok(corrections);
        }
        if stage.output.init.is_some() {
            return Err(LowerError::new(
                "online reduction correction reached unsupported accumulator",
            ));
        }

        corrections
            .into_iter()
            .map(|correction| self.propagate_online_correction(stage, input_index, correction))
            .collect()
    }

    fn propagate_online_correction(
        &self,
        stage: &Stage,
        input_index: usize,
        mut correction: OnlineCorrection,
    ) -> Result<OnlineCorrection, LowerError> {
        match (correction.kind, correction.state, stage.op, input_index) {
            (OnlineCorrectionKind::Divisor, OnlineCorrectionState::Pending, Op::Div, index)
                if index > 0 =>
            {
                correction.state = OnlineCorrectionState::Ready;
                Ok(correction)
            }
            (_, OnlineCorrectionState::Ready, _, _) => Ok(correction),
            (OnlineCorrectionKind::ExpShift, OnlineCorrectionState::Pending, Op::Sub, index)
                if index > 0 =>
            {
                correction.state = OnlineCorrectionState::Shifted;
                Ok(correction)
            }
            (OnlineCorrectionKind::ExpShift, OnlineCorrectionState::Shifted, Op::Pow, _) => {
                correction.state = OnlineCorrectionState::Ready;
                Ok(correction)
            }
            (OnlineCorrectionKind::Divisor, OnlineCorrectionState::Pending, _, _) => {
                Err(LowerError::new(
                    "online reduction correction requires a division use before accumulation",
                ))
            }
            (OnlineCorrectionKind::ExpShift, OnlineCorrectionState::Pending, _, _) => {
                Err(LowerError::new(
                    "online max correction requires a subtraction use before exponentiation",
                ))
            }
            (OnlineCorrectionKind::ExpShift, OnlineCorrectionState::Shifted, _, _) => {
                Err(LowerError::new(
                    "online max correction requires exponentiation before accumulation",
                ))
            }
            (OnlineCorrectionKind::Divisor, OnlineCorrectionState::Shifted, _, _) => Err(
                LowerError::new("division correction reached invalid shifted state"),
            ),
        }
    }

    fn validate_corrections_ready(
        &self,
        corrections: &[OnlineCorrection],
    ) -> Result<(), LowerError> {
        if corrections
            .iter()
            .all(|correction| correction.state.is_ready())
        {
            Ok(())
        } else {
            Err(LowerError::new(
                "online reduction correction reached accumulator before correction use",
            ))
        }
    }

    fn init_zero_checks(
        &self,
        stage: &Stage,
        local_loops: &[(AxisId, LoopInfo)],
        axes: &BTreeMap<AxisId, LoopInfo>,
        depth: usize,
    ) -> Result<Vec<LoopId>, LowerError> {
        let output_indexes = stage
            .output
            .shape
            .0
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let local_positions = local_loops
            .iter()
            .enumerate()
            .map(|(position, (axis, _))| (*axis, position))
            .collect::<BTreeMap<_, _>>();
        let mut checks = Vec::new();

        for (axis_index, axis) in stage.axes.iter().enumerate() {
            let axis_id = AxisId(axis_index);
            let info = axes
                .get(&axis_id)
                .ok_or_else(|| LowerError::new(format!("axis {} is not available", axis_id.0)))?;
            let index = match axis {
                Axis::Live { index, .. } | Axis::Pruned { index, .. } => *index,
            };
            if output_indexes.contains(&index) {
                continue;
            }

            let exterior = match axis {
                Axis::Live { .. } => local_positions
                    .get(&axis_id)
                    .is_some_and(|position| *position < depth),
                Axis::Pruned { .. } => true,
            };
            if exterior && !checks.contains(&info.id) {
                checks.push(info.id);
            }
        }

        Ok(checks)
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
    parent_axes: &BTreeMap<AxisId, LoopInfo>,
    axis: StageAxisRef,
) -> Result<LoopInfo, LowerError> {
    match axis {
        StageAxisRef::Local(axis) => axes
            .get(&axis)
            .cloned()
            .ok_or_else(|| LowerError::new(format!("axis {} is not available", axis.0))),
        StageAxisRef::Consumer(axis) => parent_axes
            .get(&axis)
            .cloned()
            .ok_or_else(|| LowerError::new(format!("consumer axis {} is not available", axis.0))),
    }
}

fn local_axis_depth(
    axis: StageAxisRef,
    local_positions: &BTreeMap<AxisId, usize>,
) -> Result<usize, LowerError> {
    match axis {
        StageAxisRef::Local(axis) => local_positions
            .get(&axis)
            .map(|position| position + 1)
            .ok_or_else(|| LowerError::new(format!("axis {} is not local", axis.0))),
        StageAxisRef::Consumer(_) => Ok(0),
    }
}

fn reconstruction_factors(infos: &[LoopInfo]) -> Vec<usize> {
    let factors = infos
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
        });
    normalize_reconstruction_factors(factors, infos.len())
}

fn normalize_reconstruction_factors(factors: Vec<usize>, loop_count: usize) -> Vec<usize> {
    let expected = loop_count.saturating_sub(1);
    if factors.len() > expected {
        factors[factors.len() - expected..].to_vec()
    } else {
        factors
    }
}

fn extent_level(kind: &ExtentKind) -> usize {
    match kind {
        ExtentKind::Semantic | ExtentKind::Base(_) => 0,
        ExtentKind::Split { level, .. } => *level,
    }
}

fn extent_order(kind: &ExtentKind) -> (usize, usize) {
    match kind {
        ExtentKind::Semantic | ExtentKind::Base(_) => (0, 0),
        ExtentKind::Split { level, .. } => (1, *level),
    }
}

#[derive(Clone)]
struct LoopInfo {
    id: LoopId,
    index: crate::ir::common::Index,
    kind: ExtentKind,
}

struct EmittedStage {
    block: Block,
    hoisted: Vec<HoistedBlock>,
    hoisted_corrections: Vec<HoistedCorrections>,
    corrections: Vec<OnlineCorrection>,
}

#[derive(Clone)]
struct OnlineCorrection {
    old_buffer: BufferId,
    new_buffer: BufferId,
    shape: Shape,
    kind: OnlineCorrectionKind,
    state: OnlineCorrectionState,
}

#[derive(Clone, Copy, PartialEq)]
enum OnlineCorrectionKind {
    Divisor,
    ExpShift,
}

#[derive(Clone, Copy)]
enum OnlineCorrectionState {
    Pending,
    Shifted,
    Ready,
}

impl OnlineCorrectionState {
    fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
}

struct EmittedStageKey {
    stage_index: usize,
    stage: Stage,
    inputs: Vec<Source>,
    hoisted: Vec<HoistedBlock>,
    hoisted_corrections: Vec<HoistedCorrections>,
    corrections: Vec<OnlineCorrection>,
}

#[derive(Clone)]
struct HoistedBlock {
    site: HoistSite,
    block: Block,
}

#[derive(Clone)]
struct HoistedCorrections {
    site: HoistSite,
    corrections: Vec<OnlineCorrection>,
}

enum Placement {
    Local(usize),
    Hoist(HoistSite),
}

#[derive(Clone, Copy)]
enum HoistSite {
    Before(AxisId),
    After(AxisId),
}

fn format_hoist_site(site: HoistSite) -> String {
    match site {
        HoistSite::Before(axis) => format!("consumer axis {} before site", axis.0),
        HoistSite::After(axis) => format!("consumer axis {} after site", axis.0),
    }
}

fn requires_online_reduction(stage: &Stage) -> bool {
    if stage.output.init.is_none() {
        return false;
    }
    let output_indexes = stage
        .output
        .shape
        .0
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    stage.axes.iter().any(|axis| match axis {
        Axis::Pruned { index, .. } => !output_indexes.contains(index),
        Axis::Live { .. } => false,
    })
}

fn correction_kind_key(kind: OnlineCorrectionKind) -> usize {
    match kind {
        OnlineCorrectionKind::Divisor => 0,
        OnlineCorrectionKind::ExpShift => 1,
    }
}

fn dedup_online_corrections(corrections: Vec<OnlineCorrection>) -> Vec<OnlineCorrection> {
    let mut deduped = Vec::new();
    for correction in corrections {
        if !deduped.iter().any(|existing: &OnlineCorrection| {
            existing.kind == correction.kind
                && existing.old_buffer == correction.old_buffer
                && existing.new_buffer == correction.new_buffer
        }) {
            deduped.push(correction);
        }
    }
    deduped
}

fn correction_shape_broadcasts_to_stage(correction: &OnlineCorrection, stage: &Stage) -> bool {
    let output_indexes = stage
        .output
        .shape
        .0
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    correction
        .shape
        .0
        .iter()
        .all(|index| output_indexes.contains(index))
}

fn output_uses_semantic_layout(stage: &Stage) -> bool {
    stage
        .output
        .layout
        .0
        .iter()
        .any(|dim| matches!(dim, LayoutDim::Semantic { .. }))
}

fn stage_axis_refs_for_index(
    stage: &Stage,
    index: crate::ir::common::Index,
) -> Result<Vec<StageAxisRef>, LowerError> {
    let mut refs = stage
        .axes
        .iter()
        .enumerate()
        .filter_map(|(axis, stage_axis)| {
            let (axis_index, kind, axis_ref) = match stage_axis {
                Axis::Live { index, kind } => (*index, kind, StageAxisRef::Local(AxisId(axis))),
                Axis::Pruned { index, kind, by } => (*index, kind, *by),
            };
            (axis_index == index).then_some((extent_order(kind), axis_ref))
        })
        .collect::<Vec<_>>();
    if refs.is_empty() {
        return Err(LowerError::new(format!(
            "stage has no axes for correction index {}",
            index.0
        )));
    }
    refs.sort_by_key(|(order, _)| *order);
    Ok(refs.into_iter().map(|(_, axis_ref)| axis_ref).collect())
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
    use crate::ir::kernel_program::{
        Action, BufferId, BufferKind, BufferLayout, Iter, LoopId, ScalarExpr,
    };
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::{component, front};
    use std::collections::BTreeSet;

    fn parse_component(src: &str) -> crate::ir::component::Component {
        front::parse_component(src).unwrap()
    }

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
            program.graph.nodes[0]
                .inner
                .writes
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                crate::ir::kernel_program::BufferId(2),
                crate::ir::kernel_program::BufferId(3),
            ])
        );
    }

    #[test]
    fn absorbs_root_compute_at_producer_into_consumer_kernel() {
        let component = component::expr(front::parse_expr("i+i~i||i").unwrap())
            .chain(component::expr(front::parse_expr("i-i~i||0i").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let mut ops = Vec::new();
        for node in &program.graph.nodes {
            collect_compute_ops(&node.inner.body, &mut ops);
        }

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(
            ops,
            vec![crate::ir::common::Op::Add, crate::ir::common::Op::Sub]
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
    fn emits_empty_zero_checks_for_init_before_reduction_loops() {
        let program = lower_stage_program_to_kernel_program(&lower_expr("+ijk~ij||ij!k")).unwrap();
        let init = first_init(&program.graph.nodes[0].inner.body).unwrap();
        let Action::Init { zero_checks, .. } = init else {
            unreachable!();
        };

        assert_eq!(zero_checks, &Vec::<LoopId>::new());
    }

    #[test]
    fn emits_zero_checks_for_exterior_reduction_loops() {
        let program = lower_stage_program_to_kernel_program(&lower_expr("+ijk~ij||ijk!")).unwrap();
        let init = first_init(&program.graph.nodes[0].inner.body).unwrap();
        let Action::Init { zero_checks, .. } = init else {
            unreachable!();
        };

        assert_eq!(zero_checks, &vec![LoopId(2)]);
    }

    #[test]
    fn rejects_rowwise_fused_reduction_axis_without_downstream_accumulator() {
        let mm_t = component::expr(front::parse_expr("ik*jk~ijk|i:2,j:2|iji'j'k").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,j:2|iji'j'k0").unwrap()),
        );
        let mm = component::expr(front::parse_expr("ik*kj~ijk|i:2,k:2|ik0ji'k'").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,k:2|ikji'k'0").unwrap()),
        );
        let exp = component::expr(front::parse_expr("^ij~ij|i:2,j:2|iji'j'0").unwrap());
        let row_sum = component::expr(front::parse_expr("+ij~i|i:2,j:2|iji'j'0").unwrap());
        let row_div = component::expr(front::parse_expr("ij/i~ij|i:2,j:2|iji'j'1").unwrap());
        let attention = mm_t.chain(exp).chain(mm.fanout(row_sum)).chain(row_div);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&attention).unwrap())
                .unwrap();
        let error = lower_stage_program_to_kernel_program(&stage_program).unwrap_err();

        assert_eq!(
            error.to_string(),
            "online reduction correction reached root without an accumulator"
        );
    }

    #[test]
    fn lowers_matmul_rowwise_normalize_matmul_with_shared_score_tile() {
        let scores = parse_component("ij*kj~ikj | i:2,k:2 | kii'k'j")
            .chain(parse_component("+ikj~ik | i:2,k:2 | kii'k'j0"));
        let normalize = crate::ir::component::Component::Identity
            .fanout(parse_component("+ik~i | i:2,k:2 | ki0i'k'"))
            .chain(parse_component("ik/i~ik | i:2,k:2 | ki01i'k'"));
        let matmul = parse_component("ik*kj~ijk | i:2,k:2 | ki0i'jk'")
            .chain(parse_component("+ijk~ij | i:2,k:2 | kii'jk'0"));
        let component = scores.chain(normalize).chain(matmul);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut snapshots = Vec::new();
        let mut scales = Vec::new();
        collect_online_actions(
            &program.graph.nodes[0].inner.body,
            &mut snapshots,
            &mut scales,
        );

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(scales.len(), 1);
    }

    #[test]
    fn lowers_rowwise_normalize_matmul_with_online_correction_actions() {
        let normalize = crate::ir::component::Component::Identity
            .fanout(parse_component("+ik~i | i:2,k:2 | kii'k'"))
            .chain(parse_component("ik/i~ik | i:2,k:2 | ki1i'k'"));
        let matmul = parse_component("ik*kj~ijk | i:2,k:2 | ki0i'jk'")
            .chain(parse_component("+ijk~ij | i:2,k:2 | kii'jk'0"));
        let component = normalize.chain(matmul);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut snapshots = Vec::new();
        let mut scales = Vec::new();
        collect_online_actions(
            &program.graph.nodes[0].inner.body,
            &mut snapshots,
            &mut scales,
        );

        assert_eq!(snapshots.len(), 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(
            program.buffers[snapshots[0].buffer.0].kind,
            BufferKind::Intermediate
        );
        assert_eq!(scales[0].index.len(), 2);
    }

    #[test]
    fn lowers_composed_attention_online_corrections() {
        let scores = parse_component("ij*kj~ikj | i:2,k:2 | kii'k'j")
            .chain(parse_component("+ikj~ik | i:2,k:2 | kii'k'j0"));
        let max_shift = crate::ir::component::Component::Identity
            .fanout(parse_component(">ik~i | i:2,k:2 | ki0i'k'"))
            .chain(parse_component("ik-i~ik | i:2,k:2 | ki01i'k'"));
        let exp = parse_component("^ik~ik | i:2,k:2 | ki0i'k'");
        let normalize = crate::ir::component::Component::Identity
            .fanout(parse_component("+ik~i | i:2,k:2 | ki0i'k'"))
            .chain(parse_component("ik/i~ik | i:2,k:2 | ki01i'k'"));
        let matmul = parse_component("ik*kj~ijk | i:2,k:2 | ki0i'jk'")
            .chain(parse_component("+ijk~ij | i:2,k:2 | kii'jk'0"));
        let component = scores
            .chain(max_shift)
            .chain(exp)
            .chain(normalize)
            .chain(matmul);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut scale_terms = Vec::new();
        collect_scale_terms(&program.graph.nodes[0].inner.body, &mut scale_terms);
        let exp_scales = scale_terms
            .iter()
            .filter(|factor| is_exp_shift(factor))
            .count();

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(scale_terms.len(), 3);
        assert_eq!(exp_scales, 2);
    }

    #[test]
    fn lowers_scalar_max_shift_exp_dot_with_online_correction_actions() {
        let max_shift = crate::ir::component::Component::Identity
            .pair(parse_component(">i~. | i:2 | ii'"))
            .chain(parse_component("i-.~i | i:2 | i1i'"));
        let exp = parse_component("^i~i | i:2 | i0i'");
        let dot = parse_component("i*i~i | i:2 | i0i'").chain(parse_component("+i~. | i:2 | ii'0"));
        let component = max_shift.chain(exp).chain(dot);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut snapshots = Vec::new();
        let mut scales = Vec::new();
        let mut scale_terms = Vec::new();
        collect_online_actions(
            &program.graph.nodes[0].inner.body,
            &mut snapshots,
            &mut scales,
        );
        collect_scale_terms(&program.graph.nodes[0].inner.body, &mut scale_terms);

        assert_eq!(snapshots.len(), 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(scale_terms.len(), 1);
        assert!(is_exp_shift(scale_terms[0]));
    }

    #[test]
    fn lowers_scalar_fused_reduction_axis_with_online_correction_actions() {
        let normalize = crate::ir::component::Component::Identity
            .pair(parse_component("+i~. | i:8 | ii'"))
            .chain(parse_component("i/.~i | i:8 | i1i'"));
        let dot = parse_component("i*i~i | i:8 | i0i'").chain(parse_component("+i~. | i:8 | ii'0"));
        let component = normalize.chain(dot);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut snapshots = Vec::new();
        let mut scales = Vec::new();
        collect_online_actions(
            &program.graph.nodes[0].inner.body,
            &mut snapshots,
            &mut scales,
        );

        assert_eq!(snapshots.len(), 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(
            program.buffers[snapshots[0].buffer.0].kind,
            BufferKind::Intermediate
        );
    }

    #[test]
    fn hoists_transitive_fused_producer_to_pruned_consumer_axis() {
        let mm_t = component::expr(front::parse_expr("ik*jk~ijk|i:2,j:2|jii'j'k").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,j:2|jii'j'k0").unwrap()),
        );
        let mm = component::expr(front::parse_expr("ik*kj~ijk|i:2,k:2|ki0i'k'j").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,k:2|kii'k'j0").unwrap()),
        );
        let component = mm_t.chain(mm);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let mut computes = Vec::new();
        collect_compute_paths(
            &program.graph.nodes[0].inner.body,
            &mut Vec::new(),
            &mut computes,
        );
        let output = program.outputs[0];
        let final_add_path = computes
            .iter()
            .find_map(|(op, write, path)| {
                (*op == crate::ir::common::Op::Add && *write == output).then_some(path)
            })
            .unwrap();
        let producer_add_path = computes
            .iter()
            .find_map(|(op, write, path)| {
                (*op == crate::ir::common::Op::Add && *write != output).then_some(path)
            })
            .unwrap();

        assert_eq!(&producer_add_path[..2], &final_add_path[..2]);
        assert!(!producer_add_path.starts_with(final_add_path));
    }

    #[test]
    fn hoists_nested_root_compute_before_pruned_consumer_axis() {
        let component = component::expr(front::parse_expr("^i~i||i").unwrap())
            .chain(component::expr(front::parse_expr("i*i~i||0i").unwrap()))
            .chain(component::expr(front::parse_expr("i+i~i||i0").unwrap()));
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&component).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();

        let mut computes = Vec::new();
        collect_compute_paths(
            &program.graph.nodes[0].inner.body,
            &mut Vec::new(),
            &mut computes,
        );
        let output = program.outputs[0];
        let pow_path = computes
            .iter()
            .find_map(|(op, _, path)| (*op == crate::ir::common::Op::Pow).then_some(path))
            .unwrap();
        let final_add_path = computes
            .iter()
            .find_map(|(op, write, path)| {
                (*op == crate::ir::common::Op::Add && *write == output).then_some(path)
            })
            .unwrap();

        assert_eq!(program.graph.nodes.len(), 1);
        assert!(!pow_path.starts_with(final_add_path));
    }

    #[test]
    fn shares_root_compute_at_transitive_producer_in_common_sibling_placement() {
        let mm_t = component::expr(front::parse_expr("ik*jk~ijk||ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ijk0").unwrap()));
        let exp = component::expr(front::parse_expr("^ij~ij||ij0").unwrap());
        let mm = component::expr(front::parse_expr("ij*jk~ikj||0ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ikj~ik||ijk0").unwrap()));
        let row_sum = component::expr(front::parse_expr("+ij~i||0ij").unwrap());
        let row_div = component::expr(front::parse_expr("ik/i~ik||i01k").unwrap());
        let attention = mm_t.chain(exp).chain(mm.fanout(row_sum)).chain(row_div);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&attention).unwrap())
                .unwrap();
        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut ops = Vec::new();
        for node in &program.graph.nodes {
            collect_compute_ops(&node.inner.body, &mut ops);
        }
        let mut computes = Vec::new();
        collect_compute_paths(
            &program.graph.nodes[0].inner.body,
            &mut Vec::new(),
            &mut computes,
        );
        let output = program.outputs[0];
        let pow_path = computes
            .iter()
            .find_map(|(op, _, path)| (*op == crate::ir::common::Op::Pow).then_some(path))
            .unwrap();
        let final_div_path = computes
            .iter()
            .find_map(|(op, write, path)| {
                (*op == crate::ir::common::Op::Div && *write == output).then_some(path)
            })
            .unwrap();

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(
            ops.iter()
                .filter(|op| **op == crate::ir::common::Op::Pow)
                .count(),
            1
        );
        assert!(!pow_path.starts_with(&final_div_path[..1]));
    }

    #[test]
    fn shares_non_root_transitive_compute_at() {
        let mm_t = component::expr(front::parse_expr("ik*jk~ijk|i:2,j:2|iji'j'k").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,j:2|iji'j'k0").unwrap()),
        );
        let exp = component::expr(front::parse_expr("^ij~ij|i:2,j:2|iji'j'0").unwrap());
        let mm = component::expr(front::parse_expr("ik*kj~ijk|i:2,k:2|ik0i'jk'").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,k:2|iki'jk'0").unwrap()),
        );
        let row_sum = component::expr(front::parse_expr("+ij~i|i:2,j:2|ij0i'j'").unwrap());
        let row_div = component::expr(front::parse_expr("ij/i~ij|i:2|i01i'j").unwrap());
        let attention = mm_t.chain(exp).chain(mm.fanout(row_sum)).chain(row_div);
        let stage_program =
            lower_node_graph_to_stage_program(&lower_component_to_graph(&attention).unwrap())
                .unwrap();

        let program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let mut ops = Vec::new();
        for node in &program.graph.nodes {
            collect_compute_ops(&node.inner.body, &mut ops);
        }

        assert_eq!(program.graph.nodes.len(), 1);
        assert_eq!(
            ops.iter()
                .filter(|op| **op == crate::ir::common::Op::Pow)
                .count(),
            1
        );
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
                Action::Init { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
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
                Action::Init { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
            }
        }
    }

    fn collect_online_actions<'a>(
        block: &'a crate::ir::kernel_program::Block,
        snapshots: &mut Vec<&'a crate::ir::kernel_program::Access>,
        scales: &mut Vec<&'a crate::ir::kernel_program::Access>,
    ) {
        for action in &block.0 {
            match action {
                Action::Loop { body, .. } => collect_online_actions(body, snapshots, scales),
                Action::Snapshot { write, .. } => snapshots.push(write),
                Action::Scale { write, .. } => scales.push(write),
                Action::Compute { .. } | Action::Init { .. } => {}
            }
        }
    }

    fn collect_scale_terms<'a>(
        block: &'a crate::ir::kernel_program::Block,
        scales: &mut Vec<&'a ScalarExpr>,
    ) {
        for action in &block.0 {
            match action {
                Action::Loop { body, .. } => collect_scale_terms(body, scales),
                Action::Scale { factor, .. } => scales.push(factor),
                Action::Compute { .. } | Action::Init { .. } | Action::Snapshot { .. } => {}
            }
        }
    }

    fn is_exp_shift(factor: &ScalarExpr) -> bool {
        matches!(
            factor,
            ScalarExpr::Unary {
                op: crate::ir::common::Op::Pow,
                arg,
            } if matches!(
                arg.as_ref(),
                ScalarExpr::Binary {
                    op: crate::ir::common::Op::Sub,
                    ..
                }
            )
        )
    }

    fn collect_compute_paths(
        block: &crate::ir::kernel_program::Block,
        path: &mut Vec<LoopId>,
        computes: &mut Vec<(crate::ir::common::Op, BufferId, Vec<LoopId>)>,
    ) {
        for action in &block.0 {
            match action {
                Action::Loop { id, body, .. } => {
                    path.push(*id);
                    collect_compute_paths(body, path, computes);
                    path.pop();
                }
                Action::Compute { op, write, .. } => {
                    computes.push((*op, write.buffer, path.clone()));
                }
                Action::Init { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
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
                Action::Init { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
            }
        }
        None
    }

    fn first_init(
        block: &crate::ir::kernel_program::Block,
    ) -> Option<&crate::ir::kernel_program::Action> {
        for action in &block.0 {
            match action {
                Action::Init { .. } => return Some(action),
                Action::Loop { body, .. } => {
                    if let Some(action) = first_init(body) {
                        return Some(action);
                    }
                }
                Action::Compute { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
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

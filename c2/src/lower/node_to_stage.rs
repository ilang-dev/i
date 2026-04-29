use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::graph::validate_node_graph;
use crate::check::stage::validate_stage_graph;
use crate::ir::common::{ExtentKind, Index};
use crate::ir::graph::{Graph, Node as GraphNode, NodeId, OutputId, Source};
use crate::ir::node::{AxisRef as NodeAxisRef, MultiIndex, Node};
use crate::ir::stage::{
    Axis, AxisId, AxisRef as StageAxisRef, Input as StageInput, Layout, LayoutDim,
    Output as StageOutput, ProgramShape, Shape, ShapeDim, ShapeTable, Site as StageSite, Stage,
    StageProgram,
};

pub fn lower_node_graph_to_stage_program(graph: &Graph<Node>) -> Result<StageProgram, LowerError> {
    validate_node_graph(graph).map_err(LowerError::from_node_graph)?;
    let mut builder = Builder::new(graph);
    let outputs = graph
        .outputs
        .iter()
        .copied()
        .map(|source| builder.lower_output_source(source))
        .collect::<Result<Vec<_>, _>>()?;
    let input_shapes = builder.input_shapes();
    let stage_graph = Graph {
        inputs: graph.inputs.clone(),
        nodes: builder.nodes,
        outputs,
    };
    validate_stage_graph(&stage_graph).map_err(LowerError::from_stage_graph)?;
    Ok(StageProgram {
        shapes: ShapeTable {
            inputs: input_shapes,
            nodes: builder.shapes,
        },
        graph: stage_graph,
    })
}

pub fn lower_node_graph_to_stage_graph(graph: &Graph<Node>) -> Result<Graph<Stage>, LowerError> {
    Ok(lower_node_graph_to_stage_program(graph)?.graph)
}

struct Builder<'a> {
    graph: &'a Graph<Node>,
    nodes: Vec<GraphNode<Stage>>,
    shapes: Vec<ProgramShape>,
    input_ranks: Vec<usize>,
    materialized: Vec<Option<NodeId>>,
}

impl<'a> Builder<'a> {
    fn new(graph: &'a Graph<Node>) -> Self {
        Self {
            graph,
            nodes: Vec::new(),
            shapes: Vec::new(),
            input_ranks: vec![0; graph.inputs.len()],
            materialized: vec![None; graph.nodes.len()],
        }
    }

    fn lower_output_source(&mut self, source: Source) -> Result<Source, LowerError> {
        match source {
            Source::Input(input) => Ok(Source::Input(input)),
            Source::Node(node, output) => Ok(Source::Node(self.materialized_stage(node)?, output)),
        }
    }

    fn materialized_stage(&mut self, node: NodeId) -> Result<NodeId, LowerError> {
        if let Some(stage) = self.materialized[node.0] {
            return Ok(stage);
        }
        let stage = self.instantiate_stage(node, OutputLayout::Semantic, None)?;
        self.materialized[node.0] = Some(stage);
        Ok(stage)
    }

    fn duplicate_stage(
        &mut self,
        node: NodeId,
        consumer: &Stage,
        input_index: usize,
    ) -> Result<NodeId, LowerError> {
        self.instantiate_stage(
            node,
            OutputLayout::Physical,
            Some(PruneContext {
                consumer,
                input_index,
            }),
        )
    }

    fn instantiate_stage(
        &mut self,
        node_id: NodeId,
        output_layout: OutputLayout,
        prune_context: Option<PruneContext<'_>>,
    ) -> Result<NodeId, LowerError> {
        let graph_node = &self.graph.nodes[node_id.0];
        let axis_sources = axis_sources(&graph_node.inner);
        let mut stage = lower_node_to_stage_skeleton(&graph_node.inner);

        if let Some(context) = prune_context {
            prune_producer_for_consumer(&mut stage, context.consumer, context.input_index)?;
        }

        assign_layouts(&mut stage, &graph_node.inner, &axis_sources, output_layout)?;

        let inputs = self.lower_stage_inputs(node_id, &stage)?;
        let shape = self.lower_stage_shape(&stage, inputs.as_slice())?;
        let stage_id = NodeId(self.nodes.len());
        self.nodes.push(GraphNode {
            inner: stage,
            inputs,
            outputs: graph_node.outputs.clone(),
        });
        self.shapes.push(shape);
        Ok(stage_id)
    }

    fn lower_stage_inputs(
        &mut self,
        node_id: NodeId,
        stage: &Stage,
    ) -> Result<Vec<Source>, LowerError> {
        self.graph.nodes[node_id.0]
            .inputs
            .iter()
            .copied()
            .enumerate()
            .map(|(input_index, source)| match source {
                Source::Input(input) => {
                    if stage.inputs[input_index].compute.is_some() {
                        return Err(LowerError::new(format!(
                            "node {} input {} is a graph input with a compute site",
                            node_id.0, input_index
                        )));
                    }
                    Ok(Source::Input(input))
                }
                Source::Node(producer, output) => {
                    if output != OutputId(0) {
                        return Err(LowerError::new(format!(
                            "node {} input {} references output {}",
                            node_id.0, input_index, output.0
                        )));
                    }
                    if stage.inputs[input_index].compute.is_some() {
                        Ok(Source::Node(
                            self.duplicate_stage(producer, stage, input_index)?,
                            OutputId(0),
                        ))
                    } else {
                        Ok(Source::Node(
                            self.materialized_stage(producer)?,
                            OutputId(0),
                        ))
                    }
                }
            })
            .collect()
    }

    fn lower_stage_shape(
        &mut self,
        stage: &Stage,
        inputs: &[Source],
    ) -> Result<ProgramShape, LowerError> {
        let mut dims = BTreeMap::new();
        for (input_index, input) in stage.inputs.iter().enumerate() {
            let source = inputs[input_index];
            let source_shape = self.source_shape(source, input.shape.0.len())?;
            for (dim, index) in input.shape.0.iter().enumerate() {
                dims.entry(index.0).or_insert(source_shape.0[dim]);
            }
        }

        stage
            .output
            .shape
            .0
            .iter()
            .map(|index| {
                dims.get(&index.0).copied().ok_or_else(|| {
                    LowerError::new(format!(
                        "stage output index {} has no input shape source",
                        index.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ProgramShape)
    }

    fn source_shape(&mut self, source: Source, rank: usize) -> Result<ProgramShape, LowerError> {
        match source {
            Source::Input(input) => {
                self.input_ranks[input.0] = self.input_ranks[input.0].max(rank);
                Ok(ProgramShape(
                    (0..rank).map(|dim| ShapeDim { input, dim }).collect(),
                ))
            }
            Source::Node(node, OutputId(0)) => {
                let shape = self.shapes.get(node.0).ok_or_else(|| {
                    LowerError::new(format!(
                        "stage input references shape of nonexistent stage {}",
                        node.0
                    ))
                })?;
                if shape.0.len() != rank {
                    return Err(LowerError::new(format!(
                        "stage input expects rank {} but source stage {} has rank {}",
                        rank,
                        node.0,
                        shape.0.len()
                    )));
                }
                Ok(shape.clone())
            }
            Source::Node(_, output) => Err(LowerError::new(format!(
                "stage input references output {}",
                output.0
            ))),
        }
    }

    fn input_shapes(&self) -> Vec<ProgramShape> {
        self.input_ranks
            .iter()
            .enumerate()
            .map(|(input, rank)| {
                ProgramShape(
                    (0..*rank)
                        .map(|dim| ShapeDim {
                            input: crate::ir::graph::InputId(input),
                            dim,
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
struct PruneContext<'a> {
    consumer: &'a Stage,
    input_index: usize,
}

#[derive(Clone, Copy)]
enum OutputLayout {
    Semantic,
    Physical,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AxisSource {
    index: Index,
    kind: ExtentKind,
}

fn lower_node_to_stage_skeleton(node: &Node) -> Stage {
    let axis_map = axis_map(node);
    Stage {
        op: node.op,
        axes: axis_sources(node)
            .into_iter()
            .map(|source| Axis::Live {
                index: source.index,
                kind: source.kind,
            })
            .collect(),
        output: StageOutput {
            shape: lower_shape(&node.output),
            layout: Layout(Vec::new()),
            init: node.init_site.map(|site| lower_site(site, &axis_map)),
        },
        inputs: node
            .inputs
            .iter()
            .zip(node.compute_sites.iter())
            .map(|(input, compute)| StageInput {
                shape: lower_shape(input),
                layout: Layout(Vec::new()),
                compute: compute.map(|site| lower_site(site, &axis_map)),
            })
            .collect(),
    }
}

fn assign_layouts(
    stage: &mut Stage,
    node: &Node,
    axis_sources: &[AxisSource],
    output_layout: OutputLayout,
) -> Result<(), LowerError> {
    stage.output.layout =
        lower_layout(stage, axis_sources, &node.output, output_layout, &[], false)?;
    for input_index in 0..stage.inputs.len() {
        let (layout, excluded_axes, include_consumer_axes) =
            if stage.inputs[input_index].compute.is_some() {
                (
                    OutputLayout::Physical,
                    consumer_axis_prefix(stage, input_index)?,
                    true,
                )
            } else {
                (OutputLayout::Semantic, Vec::new(), true)
            };
        stage.inputs[input_index].layout = lower_layout(
            stage,
            axis_sources,
            &node.inputs[input_index],
            layout,
            excluded_axes.as_slice(),
            include_consumer_axes,
        )?;
    }
    Ok(())
}

fn lower_shape(multi_index: &MultiIndex) -> Shape {
    Shape(multi_index.0.clone())
}

fn lower_layout(
    stage: &Stage,
    axis_sources: &[AxisSource],
    multi_index: &MultiIndex,
    layout: OutputLayout,
    excluded_axes: &[AxisId],
    include_consumer_axes: bool,
) -> Result<Layout, LowerError> {
    match layout {
        OutputLayout::Semantic => semantic_layout(stage, axis_sources, multi_index),
        OutputLayout::Physical => physical_layout(
            stage,
            axis_sources,
            multi_index,
            excluded_axes,
            include_consumer_axes,
        ),
    }
}

fn semantic_layout(
    stage: &Stage,
    axis_sources: &[AxisSource],
    multi_index: &MultiIndex,
) -> Result<Layout, LowerError> {
    multi_index
        .0
        .iter()
        .copied()
        .map(|index| {
            Ok(LayoutDim::Semantic {
                index,
                axes: axis_refs_for_index(stage, axis_sources, index)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Layout)
}

fn physical_layout(
    stage: &Stage,
    axis_sources: &[AxisSource],
    multi_index: &MultiIndex,
    excluded_axes: &[AxisId],
    include_consumer_axes: bool,
) -> Result<Layout, LowerError> {
    let mut dims = Vec::new();
    for index in &multi_index.0 {
        let mut refs = axis_sources
            .iter()
            .enumerate()
            .filter(|(_, source)| source.index == *index)
            .map(|(axis, source)| {
                let axis_id = AxisId(axis);
                let axis_ref = stage_axis_ref(stage, axis_id)?;
                Ok((extent_order(&source.kind), axis_id, axis_ref))
            })
            .collect::<Result<Vec<_>, LowerError>>()?;
        refs.sort_by_key(|(order, _, _)| *order);
        for (_, axis, axis_ref) in refs {
            if excluded_axes.contains(&axis) {
                continue;
            }
            if matches!(axis_ref, StageAxisRef::Consumer(_)) && !include_consumer_axes {
                continue;
            }
            dims.push(LayoutDim::Physical(axis_ref));
        }
    }
    Ok(Layout(dims))
}

fn stage_axis_ref(stage: &Stage, axis: AxisId) -> Result<StageAxisRef, LowerError> {
    match stage.axes.get(axis.0) {
        Some(Axis::Live { .. }) => Ok(StageAxisRef::Local(axis)),
        Some(Axis::Pruned { by, .. }) => Ok(*by),
        None => Err(LowerError::new(format!(
            "axis {} does not exist while lowering layout",
            axis.0
        ))),
    }
}

fn axis_refs_for_index(
    stage: &Stage,
    axis_sources: &[AxisSource],
    index: Index,
) -> Result<Vec<StageAxisRef>, LowerError> {
    let mut refs = axis_sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.index == index)
        .map(|(axis, source)| {
            let axis_id = AxisId(axis);
            let axis_ref = match stage.axes.get(axis) {
                Some(Axis::Live { .. }) => StageAxisRef::Local(axis_id),
                Some(Axis::Pruned { by, .. }) => *by,
                None => {
                    return Err(LowerError::new(format!(
                        "axis {} does not exist while lowering layout",
                        axis
                    )));
                }
            };
            Ok((extent_order(&source.kind), axis_ref))
        })
        .collect::<Result<Vec<_>, _>>()?;
    refs.sort_by_key(|(order, _)| *order);
    Ok(refs.into_iter().map(|(_, axis_ref)| axis_ref).collect())
}

fn axis_map(node: &Node) -> BTreeMap<(usize, usize), AxisId> {
    node.order
        .iter()
        .enumerate()
        .map(|(axis_id, axis_ref)| ((axis_ref.index.0, axis_ref.level), AxisId(axis_id)))
        .collect()
}

fn axis_sources(node: &Node) -> Vec<AxisSource> {
    node.order
        .iter()
        .map(|axis_ref| AxisSource {
            index: axis_ref.index,
            kind: extent_kind(node, *axis_ref),
        })
        .collect()
}

fn lower_site(
    site: crate::ir::node::Site,
    axis_map: &BTreeMap<(usize, usize), AxisId>,
) -> StageSite {
    match site {
        crate::ir::node::Site::Root => StageSite::Root,
        crate::ir::node::Site::At(axis_ref) => {
            StageSite::At(axis_map[&(axis_ref.index.0, axis_ref.level)])
        }
    }
}

fn extent_kind(node: &Node, axis_ref: NodeAxisRef) -> ExtentKind {
    let factors = node.splits[axis_ref.index.0]
        .0
        .iter()
        .map(|factor| factor.0)
        .collect::<Vec<_>>();
    if factors.is_empty() {
        ExtentKind::Semantic
    } else if axis_ref.level == 0 {
        ExtentKind::Base(factors)
    } else {
        ExtentKind::Split {
            level: axis_ref.level,
            factor: factors[axis_ref.level - 1],
        }
    }
}

pub(crate) fn prune_producer_for_consumer(
    producer: &mut Stage,
    consumer: &Stage,
    input_index: usize,
) -> Result<(), LowerError> {
    let pruning = prunable_axis_prefix(producer, consumer, input_index)?;
    apply_axis_pruning(producer, pruning.as_slice());
    Ok(())
}

pub(crate) fn prunable_axis_prefix(
    producer: &Stage,
    consumer: &Stage,
    input_index: usize,
) -> Result<Vec<AxisPrune>, LowerError> {
    let consumer_prefix = consumer_axis_prefix(consumer, input_index)?;
    let alignment = edge_axis_alignment(producer, consumer, input_index)?;
    let consumer_prefix_set = consumer_prefix.iter().copied().collect::<BTreeSet<_>>();
    let aligned_consumer_axes = alignment.values().copied().collect::<BTreeSet<_>>();
    let expected = consumer_prefix
        .iter()
        .copied()
        .filter(|axis| aligned_consumer_axes.contains(axis))
        .collect::<Vec<_>>();
    let mut actual = Vec::new();
    let mut pruned = Vec::new();

    for producer_axis in 0..producer.axes.len() {
        let producer_axis = AxisId(producer_axis);
        let Some(aligned_consumer_axis) = alignment.get(&producer_axis).copied() else {
            continue;
        };
        if !consumer_prefix_set.contains(&aligned_consumer_axis) {
            continue;
        }
        actual.push(aligned_consumer_axis);
        validate_prunable_axis_pair(producer, consumer, producer_axis, aligned_consumer_axis)?;
        pruned.push(AxisPrune {
            axis: producer_axis,
            by: StageAxisRef::Consumer(aligned_consumer_axis),
        });
    }

    if actual != expected {
        return Err(LowerError::new(format!(
            "producer axes align with consumer axes {:?}, expected {:?}",
            actual, expected
        )));
    }

    Ok(pruned)
}

pub(crate) fn consumer_axis_prefix(
    consumer: &Stage,
    input_index: usize,
) -> Result<Vec<AxisId>, LowerError> {
    match consumer
        .inputs
        .get(input_index)
        .and_then(|input| input.compute)
    {
        None => Ok(Vec::new()),
        Some(StageSite::Root) => Ok(Vec::new()),
        Some(StageSite::At(site)) => {
            let end = site.0.checked_add(1).ok_or_else(|| {
                LowerError::new(format!(
                    "consumer input {} compute site overflows",
                    input_index
                ))
            })?;
            if end > consumer.axes.len() {
                return Err(LowerError::new(format!(
                    "consumer input {} compute site references nonexistent axis {}",
                    input_index, site.0
                )));
            }
            Ok((0..end).map(AxisId).collect())
        }
    }
}

pub(crate) fn edge_axis_alignment(
    producer: &Stage,
    consumer: &Stage,
    input_index: usize,
) -> Result<BTreeMap<AxisId, AxisId>, LowerError> {
    let input = consumer.inputs.get(input_index).ok_or_else(|| {
        LowerError::new(format!(
            "consumer input {} does not exist for pruning",
            input_index
        ))
    })?;
    if producer.output.shape.0.len() != input.shape.0.len() {
        return Err(LowerError::new(format!(
            "producer output has {} shape dimensions but consumer input {} has {} shape dimensions",
            producer.output.shape.0.len(),
            input_index,
            input.shape.0.len()
        )));
    }

    let mut alignment = BTreeMap::new();
    for (producer_index, consumer_index) in producer
        .output
        .shape
        .0
        .iter()
        .copied()
        .zip(input.shape.0.iter().copied())
    {
        let producer_axes = live_axes_for_index(producer, producer_index)?;
        let consumer_axes = live_axes_for_index(consumer, consumer_index)?;
        if producer_axes.len() != consumer_axes.len() {
            return Err(LowerError::new(format!(
                "producer index {} has {} axes but consumer index {} has {} axes",
                producer_index.0,
                producer_axes.len(),
                consumer_index.0,
                consumer_axes.len()
            )));
        }
        for ((producer_axis, producer_kind), (consumer_axis, consumer_kind)) in
            producer_axes.into_iter().zip(consumer_axes)
        {
            if producer_kind != consumer_kind {
                return Err(LowerError::new(format!(
                    "producer extent kind {:?} does not match consumer extent kind {:?}",
                    producer_kind, consumer_kind
                )));
            }
            alignment.insert(producer_axis, consumer_axis);
        }
    }

    Ok(alignment)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AxisPrune {
    pub axis: AxisId,
    pub by: StageAxisRef,
}

pub(crate) fn apply_axis_pruning(producer: &mut Stage, axes: &[AxisPrune]) {
    for prune in axes {
        let Axis::Live { index, kind } = producer.axes[prune.axis.0].clone() else {
            continue;
        };
        producer.axes[prune.axis.0] = Axis::Pruned {
            index,
            kind,
            by: prune.by,
        };
    }
}

fn validate_prunable_axis_pair(
    producer: &Stage,
    consumer: &Stage,
    producer_axis: AxisId,
    consumer_axis: AxisId,
) -> Result<(), LowerError> {
    let producer_axis = live_axis(producer, producer_axis)?;
    let consumer_axis = stage_axis(consumer, consumer_axis)?;
    if producer_axis.kind != consumer_axis.kind {
        return Err(LowerError::new(format!(
            "producer extent kind {:?} does not match consumer extent kind {:?}",
            producer_axis.kind, consumer_axis.kind
        )));
    }
    Ok(())
}

fn stage_axis(stage: &Stage, axis: AxisId) -> Result<LiveAxis<'_>, LowerError> {
    match stage.axes.get(axis.0) {
        Some(Axis::Live { index, kind }) | Some(Axis::Pruned { index, kind, .. }) => {
            Ok(LiveAxis { index, kind })
        }
        None => Err(LowerError::new(format!(
            "axis {} does not exist for pruning",
            axis.0
        ))),
    }
}

fn live_axes_for_index(
    stage: &Stage,
    index: Index,
) -> Result<Vec<(AxisId, ExtentKind)>, LowerError> {
    let mut axes = stage
        .axes
        .iter()
        .enumerate()
        .filter_map(|(axis, value)| match value {
            Axis::Live {
                index: axis_index,
                kind,
            } if *axis_index == index => Some(Ok((AxisId(axis), kind.clone()))),
            Axis::Live { .. } => None,
            Axis::Pruned {
                index: axis_index,
                kind,
                ..
            } if *axis_index == index => Some(Ok((AxisId(axis), kind.clone()))),
            Axis::Pruned { .. } => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    axes.sort_by_key(|(_, kind)| extent_order(kind));
    Ok(axes)
}

fn live_axis(stage: &Stage, axis: AxisId) -> Result<LiveAxis<'_>, LowerError> {
    match stage.axes.get(axis.0) {
        Some(Axis::Live { index, kind }) => Ok(LiveAxis { index, kind }),
        Some(Axis::Pruned { .. }) => Err(LowerError::new(format!(
            "axis {} is already pruned",
            axis.0
        ))),
        None => Err(LowerError::new(format!(
            "axis {} does not exist for pruning",
            axis.0
        ))),
    }
}

fn extent_order(kind: &ExtentKind) -> (usize, usize) {
    match kind {
        ExtentKind::Semantic => (0, 0),
        ExtentKind::Base(_) => (0, 0),
        ExtentKind::Split { level, .. } => (1, *level),
    }
}

struct LiveAxis<'a> {
    #[allow(dead_code)]
    index: &'a Index,
    kind: &'a ExtentKind,
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

    fn from_node_graph(error: crate::check::graph::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_stage_graph(error: crate::check::stage::ValidationError) -> Self {
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
    use super::{
        apply_axis_pruning, consumer_axis_prefix, edge_axis_alignment,
        lower_node_graph_to_stage_graph, lower_node_graph_to_stage_program, prunable_axis_prefix,
        prune_producer_for_consumer, AxisPrune,
    };
    use crate::front::parse_expr;
    use crate::ir::common::{ExtentKind, Index, Op};
    use crate::ir::graph::{
        Graph, Input as GraphInput, InputId, Node as GraphNode, NodeId, Output as GraphOutput,
        OutputId, Source,
    };
    use crate::ir::node::{
        AxisRef as NodeAxisRef, MultiIndex, Node, Site as NodeSite, SplitFactor, SplitList,
    };
    use crate::ir::stage::{
        Axis, AxisId, AxisRef as StageAxisRef, Input as StageInput, Layout, LayoutDim,
        Output as StageOutput, ProgramShape, Shape, ShapeDim, Site as StageSite, Stage,
    };
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::{component, front};

    fn lower_expr(src: &str) -> Graph<Node> {
        lower_component_to_graph(&component::expr(parse_expr(src).unwrap())).unwrap()
    }

    fn live(index: usize, kind: ExtentKind) -> Axis {
        Axis::Live {
            index: Index(index),
            kind,
        }
    }

    fn pruned(index: usize, kind: ExtentKind, by: usize) -> Axis {
        Axis::Pruned {
            index: Index(index),
            kind,
            by: StageAxisRef::Consumer(AxisId(by)),
        }
    }

    fn semantic(index: usize, axes: Vec<usize>) -> LayoutDim {
        LayoutDim::Semantic {
            index: Index(index),
            axes: axes
                .into_iter()
                .map(|axis| StageAxisRef::Local(AxisId(axis)))
                .collect(),
        }
    }

    fn physical(axis: usize) -> LayoutDim {
        LayoutDim::Physical(StageAxisRef::Local(AxisId(axis)))
    }

    fn shape_dim(input: usize, dim: usize) -> ShapeDim {
        ShapeDim {
            input: InputId(input),
            dim,
        }
    }

    #[test]
    fn lowers_single_node_to_stage() {
        let graph = lower_node_graph_to_stage_graph(&lower_expr("ik*kj~ijk")).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 1);
        let stage = &graph.nodes[0].inner;
        assert_eq!(stage.op, Op::Mul);
        assert_eq!(
            stage.axes,
            vec![
                live(0, ExtentKind::Semantic),
                live(1, ExtentKind::Semantic),
                live(2, ExtentKind::Semantic),
            ]
        );
        assert_eq!(
            stage.inputs,
            vec![
                StageInput {
                    shape: Shape(vec![Index(0), Index(2)]),
                    layout: Layout(vec![semantic(0, vec![0]), semantic(2, vec![2])]),
                    compute: None,
                },
                StageInput {
                    shape: Shape(vec![Index(2), Index(1)]),
                    layout: Layout(vec![semantic(2, vec![2]), semantic(1, vec![1])]),
                    compute: None,
                },
            ]
        );
        assert_eq!(
            stage.output,
            StageOutput {
                shape: Shape(vec![Index(0), Index(1), Index(2)]),
                layout: Layout(vec![
                    semantic(0, vec![0]),
                    semantic(1, vec![1]),
                    semantic(2, vec![2])
                ]),
                init: None,
            }
        );
    }

    #[test]
    fn lowers_stage_program_shapes() {
        let program = lower_node_graph_to_stage_program(&lower_expr("ik*kj~ijk")).unwrap();

        assert_eq!(
            program.shapes.inputs,
            vec![
                ProgramShape(vec![shape_dim(0, 0), shape_dim(0, 1)]),
                ProgramShape(vec![shape_dim(1, 0), shape_dim(1, 1)]),
            ]
        );
        assert_eq!(
            program.shapes.nodes,
            vec![ProgramShape(vec![
                shape_dim(0, 0),
                shape_dim(1, 1),
                shape_dim(0, 1),
            ])]
        );
    }

    #[test]
    fn lowers_stage_program_shapes_for_duplicates() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let program = lower_node_graph_to_stage_program(&node_graph).unwrap();

        assert_eq!(
            program.shapes.nodes,
            vec![
                ProgramShape(vec![shape_dim(0, 0), shape_dim(1, 1), shape_dim(0, 1)]),
                ProgramShape(vec![shape_dim(0, 0), shape_dim(1, 1)]),
            ]
        );
    }

    #[test]
    fn lowers_splits_to_extent_kinds_and_semantic_layout_axes() {
        let graph =
            lower_node_graph_to_stage_graph(&lower_expr("ik*kj~ijk|i:2,k:8|iki'k'j")).unwrap();

        assert_eq!(
            graph.nodes[0].inner.axes,
            vec![
                live(0, ExtentKind::Base(vec![2])),
                live(2, ExtentKind::Base(vec![8])),
                live(
                    0,
                    ExtentKind::Split {
                        level: 1,
                        factor: 2
                    }
                ),
                live(
                    2,
                    ExtentKind::Split {
                        level: 1,
                        factor: 8
                    }
                ),
                live(1, ExtentKind::Semantic),
            ]
        );
        assert_eq!(
            graph.nodes[0].inner.output.layout,
            Layout(vec![
                semantic(0, vec![0, 2]),
                semantic(1, vec![4]),
                semantic(2, vec![1, 3])
            ])
        );
    }

    #[test]
    fn lowers_init_and_compute_sites() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let graph = lower_node_graph_to_stage_graph(&node_graph).unwrap();
        let stage = &graph.nodes[1].inner;

        assert_eq!(stage.output.init, Some(StageSite::At(AxisId(1))));
        assert_eq!(stage.inputs[0].compute, Some(StageSite::At(AxisId(1))));
    }

    #[test]
    fn lowers_non_compute_edges_to_materialized_stages() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let stage_graph = lower_node_graph_to_stage_graph(&node_graph).unwrap();

        assert_eq!(stage_graph.nodes.len(), 2);
        assert_eq!(
            stage_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert!(stage_graph.nodes[0]
            .inner
            .axes
            .iter()
            .all(|axis| !matches!(axis, Axis::Pruned { .. })));
        assert!(stage_graph.nodes[0]
            .inner
            .output
            .layout
            .0
            .iter()
            .all(|dim| matches!(dim, LayoutDim::Semantic { .. })));
    }

    #[test]
    fn duplicates_and_prunes_compute_at_producer() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let stage_graph = lower_node_graph_to_stage_graph(&node_graph).unwrap();

        assert_eq!(stage_graph.nodes.len(), 2);
        assert_eq!(
            stage_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(
            stage_graph.nodes[0].inner.axes,
            vec![
                pruned(0, ExtentKind::Semantic, 0),
                pruned(1, ExtentKind::Semantic, 1),
                live(2, ExtentKind::Semantic),
            ]
        );
        assert_eq!(
            stage_graph.nodes[0].inner.output.shape,
            Shape(vec![Index(0), Index(1), Index(2)])
        );
        assert_eq!(
            stage_graph.nodes[0].inner.output.layout,
            Layout(vec![physical(2)])
        );
        assert_eq!(
            stage_graph.nodes[1].inner.inputs[0].layout,
            Layout(vec![physical(2)])
        );
    }

    #[test]
    fn pruned_stage_semantic_input_reconstructs_from_consumer_axes() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap()));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let stage_graph = lower_node_graph_to_stage_graph(&node_graph).unwrap();

        assert_eq!(
            stage_graph.nodes[0].inner.inputs[0].layout,
            Layout(vec![
                LayoutDim::Semantic {
                    index: Index(0),
                    axes: vec![StageAxisRef::Consumer(AxisId(0))]
                },
                LayoutDim::Semantic {
                    index: Index(2),
                    axes: vec![StageAxisRef::Local(AxisId(2))]
                }
            ])
        );
    }

    #[test]
    fn preserves_materialized_producer_for_other_consumers() {
        let producer = component::expr(front::parse_expr("ik*kj~ijk").unwrap());
        let compute_consumer = component::expr(front::parse_expr("+ijk~ij||ij0k").unwrap());
        let materialized_consumer = component::expr(front::parse_expr("+ijk~ij").unwrap());
        let component = producer.chain(compute_consumer.fanout(materialized_consumer));
        let node_graph = lower_component_to_graph(&component).unwrap();
        let stage_graph = lower_node_graph_to_stage_graph(&node_graph).unwrap();

        assert_eq!(stage_graph.nodes.len(), 4);
        assert!(stage_graph.nodes.iter().any(|node| node
            .inner
            .axes
            .iter()
            .any(|axis| matches!(axis, Axis::Pruned { .. }))));
        assert!(stage_graph.nodes.iter().any(|node| node.inner.op == Op::Mul
            && node
                .inner
                .axes
                .iter()
                .all(|axis| !matches!(axis, Axis::Pruned { .. }))));
    }

    #[test]
    fn lowers_nested_fanout_compute_at_with_pruned_consumer_axes() {
        let mm_t = component::expr(front::parse_expr("ik*jk~ijk|i:2,j:2|iji'j'k").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,j:2|iji'j'k0").unwrap()),
        );
        let mm = component::expr(front::parse_expr("ik*kj~ijk|i:2,k:2|ik0ji'k'").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij|i:2,k:2|ikji'k'0").unwrap()),
        );
        let exp = component::expr(front::parse_expr("^ij~ij|i:2,j:2|iji'j'0").unwrap());
        let row_sum = component::expr(front::parse_expr("+ij~i|i:2,j:2|iji'j'0").unwrap());
        let row_div = component::expr(front::parse_expr("ij/i~ij|i:2,j:2|iji'j'01").unwrap());
        let attention = mm_t.chain(exp).chain(mm.fanout(row_sum)).chain(row_div);
        let graph = lower_component_to_graph(&attention).unwrap();

        lower_node_graph_to_stage_program(&graph).unwrap();
    }

    #[test]
    fn consumer_axis_prefix_stops_at_compute_site() {
        let consumer = Stage {
            op: Op::Add,
            axes: vec![
                live(0, ExtentKind::Semantic),
                live(1, ExtentKind::Semantic),
                live(2, ExtentKind::Semantic),
            ],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: Some(StageSite::At(AxisId(1))),
            },
            inputs: vec![StageInput {
                shape: Shape(vec![Index(0), Index(1), Index(2)]),
                layout: Layout(vec![physical(2)]),
                compute: Some(StageSite::At(AxisId(1))),
            }],
        };

        assert_eq!(
            consumer_axis_prefix(&consumer, 0).unwrap(),
            vec![AxisId(0), AxisId(1)]
        );
    }

    #[test]
    fn edge_axis_alignment_uses_shape_positions() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(1, ExtentKind::Semantic), live(0, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![1]), semantic(1, vec![0])]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: None,
            },
            inputs: vec![StageInput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![physical(0), physical(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        let alignment = edge_axis_alignment(&producer, &consumer, 0).unwrap();
        assert_eq!(alignment[&AxisId(1)], AxisId(0));
        assert_eq!(alignment[&AxisId(0)], AxisId(1));
    }

    #[test]
    fn prunable_prefix_uses_axis_alignment_not_axis_position() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(1, ExtentKind::Semantic), live(0, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![1]), semantic(1, vec![0])]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: None,
            },
            inputs: vec![StageInput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![physical(0), physical(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        let pruning = prunable_axis_prefix(&producer, &consumer, 0).unwrap();
        assert_eq!(
            pruning,
            vec![AxisPrune {
                axis: AxisId(1),
                by: StageAxisRef::Consumer(AxisId(0))
            }]
        );
    }

    #[test]
    fn prunable_prefix_rejects_mismatched_extent_kind() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Base(vec![4]))],
            output: StageOutput {
                shape: Shape(vec![Index(0)]),
                layout: Layout(vec![semantic(0, vec![0])]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0)]),
                layout: Layout(vec![semantic(0, vec![0])]),
                init: None,
            },
            inputs: vec![StageInput {
                shape: Shape(vec![Index(0)]),
                layout: Layout(vec![physical(0)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        let error = prunable_axis_prefix(&producer, &consumer, 0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "producer extent kind Base([4]) does not match consumer extent kind Semantic"
        );
    }

    #[test]
    fn apply_axis_pruning_records_consumer_axis() {
        let mut stage = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: None,
            },
            inputs: vec![],
        };

        apply_axis_pruning(
            &mut stage,
            &[AxisPrune {
                axis: AxisId(1),
                by: StageAxisRef::Consumer(AxisId(0)),
            }],
        );
        assert_eq!(
            stage.axes,
            vec![
                live(0, ExtentKind::Semantic),
                pruned(1, ExtentKind::Semantic, 0)
            ]
        );
    }

    #[test]
    fn prune_producer_for_consumer_prunes_prefix() {
        let mut producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![semantic(0, vec![0]), semantic(1, vec![1])]),
                init: None,
            },
            inputs: vec![StageInput {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![physical(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        prune_producer_for_consumer(&mut producer, &consumer, 0).unwrap();
        assert_eq!(
            producer.axes,
            vec![
                pruned(0, ExtentKind::Semantic, 0),
                live(1, ExtentKind::Semantic)
            ]
        );
    }

    #[test]
    fn root_compute_site_uses_physical_layout_without_pruning() {
        let graph = Graph {
            inputs: vec![GraphInput],
            nodes: vec![GraphNode {
                inner: Node {
                    op: Op::Add,
                    rank: 1,
                    inputs: vec![MultiIndex(vec![Index(0)])],
                    output: MultiIndex(vec![Index(0)]),
                    splits: vec![SplitList(vec![])],
                    order: vec![NodeAxisRef {
                        index: Index(0),
                        level: 0,
                    }],
                    compute_sites: vec![Some(NodeSite::Root)],
                    init_site: None,
                },
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![GraphOutput],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = lower_node_graph_to_stage_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0 input 0 is a graph input with a compute site"
        );
    }

    #[test]
    fn split_axis_with_multiple_factors_expands_in_layout_order() {
        let graph = Graph {
            inputs: vec![GraphInput],
            nodes: vec![GraphNode {
                inner: Node {
                    op: Op::Add,
                    rank: 1,
                    inputs: vec![MultiIndex(vec![Index(0)])],
                    output: MultiIndex(vec![Index(0)]),
                    splits: vec![SplitList(vec![SplitFactor(2), SplitFactor(4)])],
                    order: vec![
                        NodeAxisRef {
                            index: Index(0),
                            level: 0,
                        },
                        NodeAxisRef {
                            index: Index(0),
                            level: 1,
                        },
                        NodeAxisRef {
                            index: Index(0),
                            level: 2,
                        },
                    ],
                    compute_sites: vec![None],
                    init_site: None,
                },
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![GraphOutput],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let stage_graph = lower_node_graph_to_stage_graph(&graph).unwrap();
        assert_eq!(
            stage_graph.nodes[0].inner.output.layout,
            Layout(vec![semantic(0, vec![0, 1, 2])])
        );
    }
}

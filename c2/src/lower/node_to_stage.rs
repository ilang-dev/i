use std::collections::BTreeMap;
use std::fmt;

use crate::check::graph::validate_node_graph;
use crate::check::stage::validate_stage_graph;
use crate::ir::common::ExtentKind;
use crate::ir::graph::{Graph, Node as GraphNode, NodeId, OutputId, Source};
use crate::ir::node::{AxisRef as NodeAxisRef, MultiIndex, Node};
use crate::ir::stage::{
    Axis, AxisId, AxisMultiId, Input as StageInput, Output as StageOutput, Site as StageSite, Stage,
};

pub fn lower_node_graph_to_stage_graph(graph: &Graph<Node>) -> Result<Graph<Stage>, LowerError> {
    validate_node_graph(graph).map_err(LowerError::from_node_graph)?;
    let mut builder = Builder::new(graph);
    let outputs = graph
        .outputs
        .iter()
        .copied()
        .map(|source| builder.lower_output_source(source))
        .collect::<Result<Vec<_>, _>>()?;
    let stage_graph = Graph {
        inputs: graph.inputs.clone(),
        nodes: builder.nodes,
        outputs,
    };
    validate_stage_graph(&stage_graph).map_err(LowerError::from_stage_graph)?;
    Ok(stage_graph)
}

struct Builder<'a> {
    graph: &'a Graph<Node>,
    nodes: Vec<GraphNode<Stage>>,
    materialized: Vec<Option<NodeId>>,
}

impl<'a> Builder<'a> {
    fn new(graph: &'a Graph<Node>) -> Self {
        Self {
            graph,
            nodes: Vec::new(),
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
        let stage = self.instantiate_stage(node, None)?;
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
            Some(PruneContext {
                consumer,
                input_index,
            }),
        )
    }

    fn instantiate_stage(
        &mut self,
        node_id: NodeId,
        prune_context: Option<PruneContext<'_>>,
    ) -> Result<NodeId, LowerError> {
        let graph_node = &self.graph.nodes[node_id.0];
        let mut stage = lower_node_to_stage(&graph_node.inner);

        if let Some(context) = prune_context {
            prune_producer_for_consumer(&mut stage, context.consumer, context.input_index)?;
        }

        let inputs = self.lower_stage_inputs(node_id, &stage)?;
        let stage_id = NodeId(self.nodes.len());
        self.nodes.push(GraphNode {
            inner: stage,
            inputs,
            outputs: graph_node.outputs.clone(),
        });
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
                Source::Input(input) => Ok(Source::Input(input)),
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
}

#[derive(Clone, Copy)]
struct PruneContext<'a> {
    consumer: &'a Stage,
    input_index: usize,
}

fn lower_node_to_stage(node: &Node) -> Stage {
    let axis_map = axis_map(node);
    Stage {
        op: node.op,
        axes: node
            .order
            .iter()
            .map(|axis_ref| Axis::Live {
                index: axis_ref.index,
                kind: extent_kind(node, *axis_ref),
            })
            .collect(),
        output: StageOutput {
            axes: lower_multi_index(&node.output, &axis_map),
            init: node.init_site.map(|site| lower_site(site, &axis_map)),
        },
        inputs: node
            .inputs
            .iter()
            .zip(node.compute_sites.iter())
            .map(|(input, compute)| StageInput {
                axes: lower_multi_index(input, &axis_map),
                compute: compute.map(|site| lower_site(site, &axis_map)),
            })
            .collect(),
    }
}

fn axis_map(node: &Node) -> BTreeMap<(usize, usize), AxisId> {
    node.order
        .iter()
        .enumerate()
        .map(|(axis_id, axis_ref)| ((axis_ref.index.0, axis_ref.level), AxisId(axis_id)))
        .collect()
}

fn lower_multi_index(
    multi_index: &MultiIndex,
    axis_map: &BTreeMap<(usize, usize), AxisId>,
) -> AxisMultiId {
    AxisMultiId(
        multi_index
            .0
            .iter()
            .flat_map(|index| {
                axis_map
                    .iter()
                    .filter_map(move |((axis_index, _), axis_id)| {
                        (*axis_index == index.0).then_some(*axis_id)
                    })
            })
            .collect(),
    )
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
    let prefix = prunable_axis_prefix(producer, consumer, input_index)?;
    apply_axis_pruning(producer, prefix.as_slice());
    Ok(())
}

pub(crate) fn prunable_axis_prefix(
    producer: &Stage,
    consumer: &Stage,
    input_index: usize,
) -> Result<Vec<AxisId>, LowerError> {
    let consumer_prefix = consumer_axis_prefix(consumer, input_index)?;
    let alignment = edge_axis_alignment(producer, consumer, input_index)?;
    let mut pruned = Vec::new();

    for (position, consumer_axis) in consumer_prefix.iter().enumerate() {
        let producer_axis = AxisId(position);
        let aligned_consumer_axis = alignment.get(&producer_axis).copied().ok_or_else(|| {
            LowerError::new(format!(
                "producer axis {} is not present in output axes",
                producer_axis.0
            ))
        })?;
        if aligned_consumer_axis != *consumer_axis {
            return Err(LowerError::new(format!(
                "producer axis {} aligns with consumer axis {}, expected {}",
                producer_axis.0, aligned_consumer_axis.0, consumer_axis.0
            )));
        }
        validate_prunable_axis_pair(producer, consumer, producer_axis, *consumer_axis)?;
        pruned.push(producer_axis);
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
    if producer.output.axes.0.len() != input.axes.0.len() {
        return Err(LowerError::new(format!(
            "producer output has {} axes but consumer input {} has {} axes",
            producer.output.axes.0.len(),
            input_index,
            input.axes.0.len()
        )));
    }
    Ok(producer
        .output
        .axes
        .0
        .iter()
        .copied()
        .zip(input.axes.0.iter().copied())
        .collect())
}

pub(crate) fn apply_axis_pruning(producer: &mut Stage, axes: &[AxisId]) {
    for axis in axes {
        producer.axes[axis.0] = Axis::Pruned;
    }
}

fn validate_prunable_axis_pair(
    producer: &Stage,
    consumer: &Stage,
    producer_axis: AxisId,
    consumer_axis: AxisId,
) -> Result<(), LowerError> {
    let producer_axis = live_axis(producer, producer_axis)?;
    let consumer_axis = live_axis(consumer, consumer_axis)?;
    if producer_axis.kind != consumer_axis.kind {
        return Err(LowerError::new(format!(
            "producer extent kind {:?} does not match consumer extent kind {:?}",
            producer_axis.kind, consumer_axis.kind
        )));
    }
    Ok(())
}

fn live_axis(stage: &Stage, axis: AxisId) -> Result<LiveAxis<'_>, LowerError> {
    match stage.axes.get(axis.0) {
        Some(Axis::Live { index, kind }) => Ok(LiveAxis { index, kind }),
        Some(Axis::Pruned) => Err(LowerError::new(format!(
            "axis {} is already pruned",
            axis.0
        ))),
        None => Err(LowerError::new(format!(
            "axis {} does not exist for pruning",
            axis.0
        ))),
    }
}

struct LiveAxis<'a> {
    #[allow(dead_code)]
    index: &'a crate::ir::common::Index,
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
        lower_node_graph_to_stage_graph, prunable_axis_prefix, prune_producer_for_consumer,
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
        Axis, AxisId, AxisMultiId, Input as StageInput, Output as StageOutput, Site as StageSite,
        Stage,
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
                    axes: AxisMultiId(vec![AxisId(0), AxisId(2)]),
                    compute: None,
                },
                StageInput {
                    axes: AxisMultiId(vec![AxisId(2), AxisId(1)]),
                    compute: None,
                },
            ]
        );
        assert_eq!(
            stage.output,
            StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1), AxisId(2)]),
                init: None,
            }
        );
    }

    #[test]
    fn lowers_splits_to_extent_kinds() {
        let graph =
            lower_node_graph_to_stage_graph(&lower_expr("ik*kj~ijk|i:2,k:8|ik0i'k'1j")).unwrap();

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
            graph.nodes[0].inner.output.axes,
            AxisMultiId(vec![AxisId(0), AxisId(2), AxisId(4), AxisId(1), AxisId(3)])
        );
    }

    #[test]
    fn lowers_init_and_compute_sites() {
        let graph = lower_node_graph_to_stage_graph(&lower_expr("+ijk~ij||ij0k")).unwrap();
        let stage = &graph.nodes[0].inner;

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
            .all(|axis| !matches!(axis, Axis::Pruned)));
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
            vec![Axis::Pruned, Axis::Pruned, live(2, ExtentKind::Semantic)]
        );
        assert_eq!(
            stage_graph.nodes[1].inner.inputs[0].compute,
            Some(StageSite::At(AxisId(1)))
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
            .any(|axis| matches!(axis, Axis::Pruned))));
        assert!(stage_graph.nodes.iter().any(|node| node.inner.op == Op::Mul
            && node
                .inner
                .axes
                .iter()
                .all(|axis| !matches!(axis, Axis::Pruned))));
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
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: Some(StageSite::At(AxisId(1))),
            },
            inputs: vec![StageInput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1), AxisId(2)]),
                compute: Some(StageSite::At(AxisId(1))),
            }],
        };

        assert_eq!(
            consumer_axis_prefix(&consumer, 0).unwrap(),
            vec![AxisId(0), AxisId(1)]
        );
    }

    #[test]
    fn edge_axis_alignment_uses_access_positions() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(1), AxisId(0)]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![StageInput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        let alignment = edge_axis_alignment(&producer, &consumer, 0).unwrap();
        assert_eq!(alignment[&AxisId(1)], AxisId(0));
        assert_eq!(alignment[&AxisId(0)], AxisId(1));
    }

    #[test]
    fn prunable_prefix_rejects_mismatched_order() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(1), AxisId(0)]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![StageInput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        let error = prunable_axis_prefix(&producer, &consumer, 0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "producer axis 0 aligns with consumer axis 1, expected 0"
        );
    }

    #[test]
    fn prunable_prefix_rejects_mismatched_extent_kind() {
        let producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Base(vec![4]))],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0)]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0)]),
                init: None,
            },
            inputs: vec![StageInput {
                axes: AxisMultiId(vec![AxisId(0)]),
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
    fn apply_axis_pruning_sets_axes_to_pruned() {
        let mut stage = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![],
        };

        apply_axis_pruning(&mut stage, &[AxisId(1)]);
        assert_eq!(
            stage.axes,
            vec![live(0, ExtentKind::Semantic), Axis::Pruned]
        );
    }

    #[test]
    fn prune_producer_for_consumer_prunes_prefix() {
        let mut producer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![live(0, ExtentKind::Semantic), live(1, ExtentKind::Semantic)],
            output: StageOutput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![StageInput {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                compute: Some(StageSite::At(AxisId(0))),
            }],
        };

        prune_producer_for_consumer(&mut producer, &consumer, 0).unwrap();
        assert_eq!(
            producer.axes,
            vec![Axis::Pruned, live(1, ExtentKind::Semantic)]
        );
    }

    #[test]
    fn root_compute_site_does_not_prune() {
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

        let stage_graph = lower_node_graph_to_stage_graph(&graph).unwrap();
        assert_eq!(
            stage_graph.nodes[0].inner.inputs[0].compute,
            Some(StageSite::Root)
        );
        assert_eq!(
            stage_graph.nodes[0].inner.axes,
            vec![live(0, ExtentKind::Semantic)]
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
            stage_graph.nodes[0].inner.axes,
            vec![
                live(0, ExtentKind::Base(vec![2, 4])),
                live(
                    0,
                    ExtentKind::Split {
                        level: 1,
                        factor: 2
                    }
                ),
                live(
                    0,
                    ExtentKind::Split {
                        level: 2,
                        factor: 4
                    }
                ),
            ]
        );
        assert_eq!(
            stage_graph.nodes[0].inner.output.axes,
            AxisMultiId(vec![AxisId(0), AxisId(1), AxisId(2)])
        );
    }
}

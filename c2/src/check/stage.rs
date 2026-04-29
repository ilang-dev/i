use std::collections::BTreeSet;
use std::fmt;

use crate::check::graph::validate_graph;
use crate::ir::common::Index;
use crate::ir::graph::{Graph, NodeId, OutputId, Source};
use crate::ir::stage::{Axis, AxisId, AxisRef, Layout, LayoutDim, Shape, Site, Stage};

pub fn validate_stage(stage: &Stage) -> Result<(), ValidationError> {
    validate_shape(&stage.output.shape).map_err(|message| err(format!("output {}", message)))?;
    validate_layout(stage, &stage.output.shape, &stage.output.layout)
        .map_err(|message| err(format!("output {}", message)))?;
    validate_site(stage, stage.output.init)
        .map_err(|message| err(format!("output {}", message)))?;

    for (axis_index, axis) in stage.axes.iter().enumerate() {
        validate_axis(stage, axis)
            .map_err(|message| err(format!("axis {} {}", axis_index, message)))?;
    }

    for (input_index, input) in stage.inputs.iter().enumerate() {
        validate_shape(&input.shape)
            .map_err(|message| err(format!("input {} {}", input_index, message)))?;
        validate_layout(stage, &input.shape, &input.layout)
            .map_err(|message| err(format!("input {} {}", input_index, message)))?;
        validate_site(stage, input.compute)
            .map_err(|message| err(format!("input {} {}", input_index, message)))?;
    }

    let output_indexes = stage
        .output
        .shape
        .0
        .iter()
        .map(|index| index.0)
        .collect::<BTreeSet<_>>();
    let has_reduction = stage.axes.iter().any(|axis| match axis {
        Axis::Live { index, .. } => !output_indexes.contains(&index.0),
        Axis::Pruned { .. } => false,
    });
    match (has_reduction, stage.output.init) {
        (false, None) => Ok(()),
        (false, Some(_)) => Err(err("pointwise stage cannot have an init site")),
        (true, None) => Err(err("reduction stage must have an init site")),
        (true, Some(_)) => Ok(()),
    }
}

pub fn validate_stage_graph(graph: &Graph<Stage>) -> Result<(), ValidationError> {
    validate_graph(graph, |stage| {
        validate_stage(stage).map_err(|error| error.to_string())
    })
    .map_err(|error| err(error.to_string()))?;

    for (node_index, node) in graph.nodes.iter().enumerate() {
        if node.inputs.len() != node.inner.inputs.len() {
            return Err(err(format!(
                "node {}: graph has {} inputs for {} stage inputs",
                node_index,
                node.inputs.len(),
                node.inner.inputs.len()
            )));
        }
        if node.outputs.len() != 1 {
            return Err(err(format!(
                "node {}: graph has {} outputs for stage",
                node_index,
                node.outputs.len()
            )));
        }
    }

    validate_consumer_axis_refs(graph)
}

fn validate_consumer_axis_refs(graph: &Graph<Stage>) -> Result<(), ValidationError> {
    let consumers = stage_consumers(graph);

    for (node_index, node) in graph.nodes.iter().enumerate() {
        let consumer_refs = consumer_axis_refs(&node.inner);
        if consumer_refs.is_empty() {
            continue;
        }

        if graph
            .outputs
            .iter()
            .any(|source| *source == Source::Node(NodeId(node_index), OutputId(0)))
        {
            return Err(err(format!(
                "node {}: graph output references consumer-scoped stage",
                node_index
            )));
        }

        let node_consumers = &consumers[node_index];
        if node_consumers.len() != 1 {
            return Err(err(format!(
                "node {}: consumer-scoped stage must have exactly one consumer, found {}",
                node_index,
                node_consumers.len()
            )));
        }

        let (consumer_index, input_index) = node_consumers[0];
        let consumer = &graph.nodes[consumer_index].inner;
        if consumer.inputs[input_index].compute.is_none() {
            return Err(err(format!(
                "node {}: consumer-scoped stage consumer {} input {} has no compute site",
                node_index, consumer_index, input_index
            )));
        }

        for axis in consumer_refs {
            if axis.0 >= consumer.axes.len() {
                return Err(err(format!(
                    "node {}: references nonexistent consumer axis {}",
                    node_index, axis.0
                )));
            }
        }
    }

    Ok(())
}

fn stage_consumers(graph: &Graph<Stage>) -> Vec<Vec<(usize, usize)>> {
    let mut consumers = vec![Vec::new(); graph.nodes.len()];
    for (consumer_index, node) in graph.nodes.iter().enumerate() {
        for (input_index, source) in node.inputs.iter().enumerate() {
            if let Source::Node(NodeId(producer_index), OutputId(0)) = *source {
                consumers[producer_index].push((consumer_index, input_index));
            }
        }
    }
    consumers
}

fn consumer_axis_refs(stage: &Stage) -> BTreeSet<AxisId> {
    let mut refs = BTreeSet::new();
    for axis in &stage.axes {
        if let Axis::Pruned {
            by: AxisRef::Consumer(axis),
            ..
        } = axis
        {
            refs.insert(*axis);
        }
    }
    collect_consumer_refs_from_layout(&stage.output.layout, &mut refs);
    for input in &stage.inputs {
        collect_consumer_refs_from_layout(&input.layout, &mut refs);
    }
    refs
}

fn collect_consumer_refs_from_layout(layout: &Layout, refs: &mut BTreeSet<AxisId>) {
    for dim in &layout.0 {
        match dim {
            LayoutDim::Physical(AxisRef::Consumer(axis)) => {
                refs.insert(*axis);
            }
            LayoutDim::Physical(AxisRef::Local(_)) => {}
            LayoutDim::Semantic { axes, .. } => {
                for axis in axes {
                    if let AxisRef::Consumer(axis) = axis {
                        refs.insert(*axis);
                    }
                }
            }
        }
    }
}

fn validate_axis(stage: &Stage, axis: &Axis) -> Result<(), String> {
    match axis {
        Axis::Live { .. } => Ok(()),
        Axis::Pruned {
            by: AxisRef::Consumer(_),
            ..
        } => Ok(()),
        Axis::Pruned {
            by: AxisRef::Local(axis),
            ..
        } => {
            validate_axis_id(stage, *axis)?;
            Err(format!("is pruned by local axis {}", axis.0))
        }
    }
}

fn validate_shape(shape: &Shape) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for index in &shape.0 {
        if !seen.insert(index.0) {
            return Err(format!("shape repeats index {}", index.0));
        }
    }
    Ok(())
}

fn validate_layout(stage: &Stage, shape: &Shape, layout: &Layout) -> Result<(), String> {
    for dim in &layout.0 {
        validate_layout_dim(stage, shape, dim)?;
    }
    validate_semantic_layout_order(shape, layout)
}

fn validate_layout_dim(stage: &Stage, shape: &Shape, dim: &LayoutDim) -> Result<(), String> {
    match dim {
        LayoutDim::Physical(axis) => validate_axis_ref(stage, *axis),
        LayoutDim::Semantic { index, axes } => {
            validate_shape_index(shape, *index)?;
            if axes.is_empty() {
                return Err(format!("semantic dimension {} has no axes", index.0));
            }
            let mut seen = BTreeSet::new();
            for axis in axes {
                validate_axis_ref(stage, *axis)?;
                if !seen.insert(axis_key(*axis)) {
                    return Err(format!("semantic dimension {} repeats axis", index.0));
                }
            }
            Ok(())
        }
    }
}

fn validate_semantic_layout_order(shape: &Shape, layout: &Layout) -> Result<(), String> {
    let semantic_indexes = layout
        .0
        .iter()
        .filter_map(|dim| match dim {
            LayoutDim::Semantic { index, .. } => Some(*index),
            LayoutDim::Physical(_) => None,
        })
        .collect::<Vec<_>>();
    if semantic_indexes.is_empty() {
        return Ok(());
    }
    if semantic_indexes != shape.0 {
        return Err(format!(
            "semantic layout indexes are {}, expected {}",
            format_indexes(&semantic_indexes),
            format_indexes(&shape.0)
        ));
    }
    Ok(())
}

fn validate_shape_index(shape: &Shape, index: Index) -> Result<(), String> {
    if !shape.0.contains(&index) {
        return Err(format!("references index {} outside shape", index.0));
    }
    Ok(())
}

fn validate_site(stage: &Stage, site: Option<Site>) -> Result<(), String> {
    match site {
        None | Some(Site::Root) => Ok(()),
        Some(Site::At(axis)) => validate_axis_id(stage, axis),
    }
}

fn validate_axis_ref(stage: &Stage, axis_ref: AxisRef) -> Result<(), String> {
    match axis_ref {
        AxisRef::Local(axis) => validate_axis_id(stage, axis),
        AxisRef::Consumer(_) => Ok(()),
    }
}

fn validate_axis_id(stage: &Stage, axis: AxisId) -> Result<(), String> {
    if axis.0 >= stage.axes.len() {
        return Err(format!("references nonexistent axis {}", axis.0));
    }
    Ok(())
}

fn axis_key(axis_ref: AxisRef) -> (usize, usize) {
    match axis_ref {
        AxisRef::Local(axis) => (0, axis.0),
        AxisRef::Consumer(axis) => (1, axis.0),
    }
}

fn format_indexes(indexes: &[Index]) -> String {
    indexes
        .iter()
        .map(|index| index.0.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn err(message: impl Into<String>) -> ValidationError {
    ValidationError {
        message: message.into(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::{validate_stage, validate_stage_graph};
    use crate::ir::common::{ExtentKind, Index, Op};
    use crate::ir::graph::{
        Graph, Input as GraphInput, Node, NodeId, Output as GraphOutput, OutputId, Source,
    };
    use crate::ir::stage::{
        Axis, AxisId, AxisRef, Input, Layout, LayoutDim, Output, Shape, Site, Stage,
    };

    fn pointwise_stage() -> Stage {
        Stage {
            op: Op::Add,
            axes: vec![
                Axis::Live {
                    index: Index(0),
                    kind: ExtentKind::Semantic,
                },
                Axis::Live {
                    index: Index(1),
                    kind: ExtentKind::Semantic,
                },
            ],
            output: Output {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![
                    LayoutDim::Semantic {
                        index: Index(0),
                        axes: vec![AxisRef::Local(AxisId(0))],
                    },
                    LayoutDim::Semantic {
                        index: Index(1),
                        axes: vec![AxisRef::Local(AxisId(1))],
                    },
                ]),
                init: None,
            },
            inputs: vec![Input {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![
                    LayoutDim::Semantic {
                        index: Index(0),
                        axes: vec![AxisRef::Local(AxisId(0))],
                    },
                    LayoutDim::Semantic {
                        index: Index(1),
                        axes: vec![AxisRef::Local(AxisId(1))],
                    },
                ]),
                compute: None,
            }],
        }
    }

    #[test]
    fn accepts_pointwise_stage() {
        assert!(validate_stage(&pointwise_stage()).is_ok());
    }

    #[test]
    fn rejects_repeated_shape_index() {
        let mut stage = pointwise_stage();
        stage.inputs[0].shape = Shape(vec![Index(0), Index(0)]);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(error.to_string(), "input 0 shape repeats index 0");
    }

    #[test]
    fn rejects_invalid_output_axis() {
        let mut stage = pointwise_stage();
        stage.output.layout = Layout(vec![LayoutDim::Physical(AxisRef::Local(AxisId(2)))]);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(error.to_string(), "output references nonexistent axis 2");
    }

    #[test]
    fn rejects_semantic_layout_out_of_shape_order() {
        let mut stage = pointwise_stage();
        stage.output.layout = Layout(vec![
            LayoutDim::Semantic {
                index: Index(1),
                axes: vec![AxisRef::Local(AxisId(1))],
            },
            LayoutDim::Semantic {
                index: Index(0),
                axes: vec![AxisRef::Local(AxisId(0))],
            },
        ]);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(
            error.to_string(),
            "output semantic layout indexes are 1,0, expected 0,1"
        );
    }

    #[test]
    fn rejects_pointwise_init_site() {
        let mut stage = pointwise_stage();
        stage.output.init = Some(Site::Root);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(
            error.to_string(),
            "pointwise stage cannot have an init site"
        );
    }

    #[test]
    fn rejects_reduction_without_init_site() {
        let mut stage = pointwise_stage();
        stage.axes.push(Axis::Live {
            index: Index(2),
            kind: ExtentKind::Semantic,
        });

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(error.to_string(), "reduction stage must have an init site");
    }

    #[test]
    fn accepts_pruned_stage_with_compute_consumer() {
        let producer = Stage {
            op: Op::Mul,
            axes: vec![
                Axis::Pruned {
                    index: Index(0),
                    kind: ExtentKind::Semantic,
                    by: AxisRef::Consumer(AxisId(0)),
                },
                Axis::Live {
                    index: Index(1),
                    kind: ExtentKind::Semantic,
                },
            ],
            output: Output {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![LayoutDim::Physical(AxisRef::Local(AxisId(1)))]),
                init: None,
            },
            inputs: vec![],
        };
        let consumer = Stage {
            op: Op::Add,
            axes: vec![
                Axis::Live {
                    index: Index(0),
                    kind: ExtentKind::Semantic,
                },
                Axis::Live {
                    index: Index(1),
                    kind: ExtentKind::Semantic,
                },
            ],
            output: Output {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![
                    LayoutDim::Semantic {
                        index: Index(0),
                        axes: vec![AxisRef::Local(AxisId(0))],
                    },
                    LayoutDim::Semantic {
                        index: Index(1),
                        axes: vec![AxisRef::Local(AxisId(1))],
                    },
                ]),
                init: None,
            },
            inputs: vec![Input {
                shape: Shape(vec![Index(0), Index(1)]),
                layout: Layout(vec![LayoutDim::Physical(AxisRef::Local(AxisId(1)))]),
                compute: Some(Site::At(AxisId(0))),
            }],
        };
        let graph = Graph {
            inputs: vec![],
            nodes: vec![
                Node {
                    inner: producer,
                    inputs: vec![],
                    outputs: vec![GraphOutput],
                },
                Node {
                    inner: consumer,
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![GraphOutput],
                },
            ],
            outputs: vec![Source::Node(NodeId(1), OutputId(0))],
        };

        assert!(validate_stage_graph(&graph).is_ok());
    }

    #[test]
    fn rejects_consumer_scoped_graph_output() {
        let mut stage = pointwise_stage();
        stage.axes[0] = Axis::Pruned {
            index: Index(0),
            kind: ExtentKind::Semantic,
            by: AxisRef::Consumer(AxisId(0)),
        };
        let graph = Graph {
            inputs: vec![GraphInput],
            nodes: vec![Node {
                inner: stage,
                inputs: vec![Source::Input(crate::ir::graph::InputId(0))],
                outputs: vec![GraphOutput],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_stage_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: graph output references consumer-scoped stage"
        );
    }
}

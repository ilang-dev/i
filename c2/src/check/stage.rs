use std::collections::BTreeSet;
use std::fmt;

use crate::check::graph::validate_graph;
use crate::ir::graph::{Graph, NodeId, OutputId, Source};
use crate::ir::stage::{Axis, AxisId, AxisMultiId, Site, Stage};

pub fn validate_stage(stage: &Stage) -> Result<(), ValidationError> {
    validate_axis_multi_id(stage, &stage.output.axes)
        .map_err(|message| err(format!("output {}", message)))?;
    validate_site(stage, stage.output.init)
        .map_err(|message| err(format!("output {}", message)))?;

    for (input_index, input) in stage.inputs.iter().enumerate() {
        validate_axis_multi_id(stage, &input.axes)
            .map_err(|message| err(format!("input {} {}", input_index, message)))?;
        validate_site(stage, input.compute)
            .map_err(|message| err(format!("input {} {}", input_index, message)))?;
    }

    let has_reduction = stage
        .axes
        .iter()
        .enumerate()
        .any(|(axis, _)| !stage.output.axes.0.contains(&AxisId(axis)));
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

    validate_pruned_axes(graph)
}

fn validate_pruned_axes(graph: &Graph<Stage>) -> Result<(), ValidationError> {
    let consumers = stage_consumers(graph);

    for (node_index, node) in graph.nodes.iter().enumerate() {
        let pruned_axes = node
            .inner
            .axes
            .iter()
            .enumerate()
            .filter_map(|(axis_index, axis)| {
                matches!(axis, Axis::Pruned).then_some(AxisId(axis_index))
            })
            .collect::<Vec<_>>();
        if pruned_axes.is_empty() {
            continue;
        }

        if graph
            .outputs
            .iter()
            .any(|source| *source == Source::Node(NodeId(node_index), OutputId(0)))
        {
            return Err(err(format!(
                "node {}: graph output references pruned stage",
                node_index
            )));
        }

        let node_consumers = &consumers[node_index];
        if node_consumers.len() != 1 {
            return Err(err(format!(
                "node {}: pruned stage must have exactly one consumer, found {}",
                node_index,
                node_consumers.len()
            )));
        }

        let (consumer_index, input_index) = node_consumers[0];
        let consumer = &graph.nodes[consumer_index].inner;
        if consumer.inputs[input_index].compute.is_none() {
            return Err(err(format!(
                "node {}: pruned stage consumer {} input {} has no compute site",
                node_index, consumer_index, input_index
            )));
        }

        for axis in pruned_axes {
            resolve_pruned_axis(&node.inner, consumer, input_index, axis).map_err(|message| {
                err(format!("node {} axis {} {}", node_index, axis.0, message))
            })?;
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

fn resolve_pruned_axis(
    producer: &Stage,
    consumer: &Stage,
    input_index: usize,
    axis: AxisId,
) -> Result<AxisId, String> {
    let position = producer
        .output
        .axes
        .0
        .iter()
        .position(|candidate| *candidate == axis)
        .ok_or_else(|| "is not present in output axes".to_string())?;
    consumer
        .inputs
        .get(input_index)
        .and_then(|input| input.axes.0.get(position))
        .copied()
        .ok_or_else(|| "does not resolve through consumer input axes".to_string())
}

fn validate_axis_multi_id(stage: &Stage, axes: &AxisMultiId) -> Result<(), String> {
    for axis in &axes.0 {
        validate_axis_id(stage, *axis)?;
    }
    let mut seen = BTreeSet::new();
    for axis in &axes.0 {
        if !seen.insert(axis.0) {
            return Err(format!("repeats axis {}", axis.0));
        }
    }
    Ok(())
}

fn validate_site(stage: &Stage, site: Option<Site>) -> Result<(), String> {
    match site {
        None | Some(Site::Root) => Ok(()),
        Some(Site::At(axis)) => validate_axis_id(stage, axis),
    }
}

fn validate_axis_id(stage: &Stage, axis: AxisId) -> Result<(), String> {
    if axis.0 >= stage.axes.len() {
        return Err(format!("references nonexistent axis {}", axis.0));
    }
    Ok(())
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
    use crate::ir::stage::{Axis, AxisId, AxisMultiId, Input, Output, Site, Stage};

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
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![Input {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                compute: None,
            }],
        }
    }

    #[test]
    fn accepts_pointwise_stage() {
        assert!(validate_stage(&pointwise_stage()).is_ok());
    }

    #[test]
    fn rejects_invalid_output_axis() {
        let mut stage = pointwise_stage();
        stage.output.axes = AxisMultiId(vec![AxisId(2)]);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(error.to_string(), "output references nonexistent axis 2");
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
        stage.output.axes = AxisMultiId(vec![AxisId(0)]);

        let error = validate_stage(&stage).unwrap_err();
        assert_eq!(error.to_string(), "reduction stage must have an init site");
    }

    #[test]
    fn accepts_pruned_stage_with_compute_consumer() {
        let producer = Stage {
            op: Op::Mul,
            axes: vec![
                Axis::Pruned,
                Axis::Live {
                    index: Index(1),
                    kind: ExtentKind::Semantic,
                },
            ],
            output: Output {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
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
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
                init: None,
            },
            inputs: vec![Input {
                axes: AxisMultiId(vec![AxisId(0), AxisId(1)]),
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
    fn rejects_pruned_graph_output() {
        let mut stage = pointwise_stage();
        stage.axes[0] = Axis::Pruned;
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
            "node 0: graph output references pruned stage"
        );
    }
}

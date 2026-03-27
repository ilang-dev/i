use std::fmt;

use crate::check::stage::validate_scheduled_stage;
use crate::ir::graph::{Graph, InputId, NodeId, OutputId, Source};
use crate::ir::stage::ScheduledStage;

pub fn validate_graph<T, F>(graph: &Graph<T>, mut validate_node: F) -> Result<(), ValidationError>
where
    F: FnMut(&T) -> Result<(), String>,
{
    for (node_index, node) in graph.nodes.iter().enumerate() {
        validate_node(&node.inner)
            .map_err(|message| err(format!("node {}: {}", node_index, message)))?;
        for source in &node.inputs {
            validate_node_source(graph, node_index, *source)?;
        }
    }

    for source in &graph.outputs {
        validate_output_source(graph, *source)?;
    }

    Ok(())
}

pub fn validate_scheduled_stage_graph(
    graph: &Graph<ScheduledStage>,
) -> Result<(), ValidationError> {
    validate_graph(graph, |stage| {
        validate_scheduled_stage(stage).map_err(|error| error.to_string())
    })
}

fn validate_node_source<T>(
    graph: &Graph<T>,
    node_index: usize,
    source: Source,
) -> Result<(), ValidationError> {
    match source {
        Source::Input(InputId(index)) => {
            if index >= graph.inputs.len() {
                return Err(err(format!(
                    "node {} input references nonexistent input {}",
                    node_index, index
                )));
            }
        }
        Source::Node(NodeId(index), OutputId(output)) => {
            if index >= graph.nodes.len() {
                return Err(err(format!(
                    "node {} input references nonexistent node {}",
                    node_index, index
                )));
            }
            if index >= node_index {
                return Err(err(format!(
                    "node {} input references non-prior node {}",
                    node_index, index
                )));
            }
            if output >= graph.nodes[index].outputs.len() {
                return Err(err(format!(
                    "node {} input references nonexistent output {} of node {}",
                    node_index, output, index
                )));
            }
        }
    }

    Ok(())
}

fn validate_output_source<T>(graph: &Graph<T>, source: Source) -> Result<(), ValidationError> {
    match source {
        Source::Input(InputId(index)) => {
            if index >= graph.inputs.len() {
                return Err(err(format!(
                    "output references nonexistent input {}",
                    index
                )));
            }
        }
        Source::Node(NodeId(index), OutputId(output)) => {
            if index >= graph.nodes.len() {
                return Err(err(format!("output references nonexistent node {}", index)));
            }
            if output >= graph.nodes[index].outputs.len() {
                return Err(err(format!(
                    "output references nonexistent output {} of node {}",
                    output, index
                )));
            }
        }
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
    use super::{validate_graph, validate_scheduled_stage_graph};
    use crate::ir::common::Op;
    use crate::ir::graph::{Graph, Input, InputId, Node, NodeId, Output, OutputId, Source};
    use crate::ir::stage::{Axis, AxisRef, Index, Schedule, ScheduledStage, SplitList, Stage};

    #[test]
    fn accepts_valid_graph() {
        let graph = Graph {
            inputs: vec![Input, Input],
            nodes: vec![
                Node {
                    inner: 1usize,
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: 2usize,
                    inputs: vec![
                        Source::Node(NodeId(0), OutputId(0)),
                        Source::Input(InputId(1)),
                    ],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(1), OutputId(0))],
        };

        assert!(validate_graph(&graph, |_| Ok(())).is_ok());
    }

    #[test]
    fn rejects_nonexistent_node_input() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: 1usize,
                inputs: vec![Source::Node(NodeId(1), OutputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_graph(&graph, |_| Ok(())).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0 input references nonexistent node 1"
        );
    }

    #[test]
    fn rejects_forward_node_input() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: 1usize,
                    inputs: vec![Source::Node(NodeId(1), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: 2usize,
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_graph(&graph, |_| Ok(())).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0 input references non-prior node 1"
        );
    }

    #[test]
    fn rejects_nonexistent_output_node() {
        let graph = Graph::<usize> {
            inputs: vec![Input],
            nodes: vec![],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_graph(&graph, |_| Ok(())).unwrap_err();
        assert_eq!(error.to_string(), "output references nonexistent node 0");
    }

    #[test]
    fn rejects_invalid_scheduled_stage_node() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: ScheduledStage {
                    stage: Stage {
                        op: Op::Add,
                        rank: 1,
                        inputs: vec![Index(vec![Axis(0)])],
                        output: Index(vec![Axis(0)]),
                    },
                    schedule: Schedule {
                        splits: vec![SplitList(vec![])],
                        order: vec![AxisRef {
                            axis: Axis(0),
                            part: 0,
                        }],
                        compute_sites: vec![None],
                        init_site: Some(crate::ir::stage::Site::Root),
                    },
                },
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_scheduled_stage_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: pointwise stage cannot have an init site"
        );
    }
}

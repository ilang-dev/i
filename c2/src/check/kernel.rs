use std::fmt;

use crate::check::graph::{validate_graph, validate_scheduled_stage_graph};
use crate::ir::graph::{Graph, NodeId, OutputId, Source};
use crate::ir::kernel::Kernel;

pub fn validate_kernel(kernel: &Kernel) -> Result<(), ValidationError> {
    validate_scheduled_stage_graph(&kernel.0).map_err(|error| err(error.to_string()))?;

    if kernel.0.outputs.len() != kernel.0.nodes.len() {
        return Err(err(format!(
            "kernel has {} outputs for {} nodes",
            kernel.0.outputs.len(),
            kernel.0.nodes.len()
        )));
    }

    for (node_index, node) in kernel.0.nodes.iter().enumerate() {
        for (input_index, source) in node.inputs.iter().enumerate() {
            let compute_site = node.inner.schedule.compute_sites[input_index];
            match (source, compute_site) {
                (Source::Input(_), None) => {}
                (Source::Node(_, _), Some(_)) => {}
                (Source::Input(_), Some(_)) => {
                    return Err(err(format!(
                        "node {} input {} is sourced from kernel input but has a compute site",
                        node_index, input_index
                    )))
                }
                (Source::Node(_, _), None) => {
                    return Err(err(format!(
                        "node {} input {} is sourced from kernel node but has no compute site",
                        node_index, input_index
                    )))
                }
            }
        }

        if kernel.0.outputs[node_index] != Source::Node(NodeId(node_index), OutputId(0)) {
            return Err(err(format!(
                "kernel output {} must reference node {} output 0",
                node_index, node_index
            )));
        }
    }

    Ok(())
}

pub fn validate_kernel_graph(graph: &Graph<Kernel>) -> Result<(), ValidationError> {
    validate_graph(graph, |kernel| {
        validate_kernel(kernel).map_err(|error| error.to_string())
    })
    .map_err(|error| err(error.to_string()))?;

    for (node_index, node) in graph.nodes.iter().enumerate() {
        if node.inputs.len() != node.inner.0.inputs.len() {
            return Err(err(format!(
                "node {}: graph has {} inputs for {} kernel inputs",
                node_index,
                node.inputs.len(),
                node.inner.0.inputs.len()
            )));
        }
        if node.outputs.len() != node.inner.0.outputs.len() {
            return Err(err(format!(
                "node {}: graph has {} outputs for {} kernel outputs",
                node_index,
                node.outputs.len(),
                node.inner.0.outputs.len()
            )));
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
    use super::{validate_kernel, validate_kernel_graph};
    use crate::ir::common::Op;
    use crate::ir::graph::{Graph, Input, InputId, Node, NodeId, Output, OutputId, Source};
    use crate::ir::kernel::Kernel;
    use crate::ir::stage::{
        Axis, AxisRef, Index, Schedule, ScheduledStage, Site, SplitList, Stage,
    };

    fn stage(input_count: usize, compute_sites: Vec<Option<Site>>) -> ScheduledStage {
        ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 1,
                inputs: (0..input_count).map(|_| Index(vec![Axis(0)])).collect(),
                output: Index(vec![Axis(0)]),
            },
            schedule: Schedule {
                splits: vec![SplitList(vec![])],
                order: vec![AxisRef {
                    axis: Axis(0),
                    part: 0,
                }],
                compute_sites,
                init_site: None,
            },
        }
    }

    #[test]
    fn accepts_valid_kernel() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ],
        });

        assert!(validate_kernel(&kernel).is_ok());
    }

    #[test]
    fn rejects_kernel_input_with_compute_site() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: stage(1, vec![Some(Site::Root)]),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        });

        let error = validate_kernel(&kernel).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0 input 0 is sourced from kernel input but has a compute site"
        );
    }

    #[test]
    fn rejects_kernel_node_source_without_compute_site() {
        let kernel = Kernel(Graph {
            inputs: vec![],
            nodes: vec![
                Node {
                    inner: stage(0, vec![]),
                    inputs: vec![],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ],
        });

        let error = validate_kernel(&kernel).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 1 input 0 is sourced from kernel node but has no compute site"
        );
    }

    #[test]
    fn rejects_noncanonical_kernel_outputs() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: stage(1, vec![None]),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Input(InputId(0))],
        });

        let error = validate_kernel(&kernel).unwrap_err();
        assert_eq!(
            error.to_string(),
            "kernel output 0 must reference node 0 output 0"
        );
    }

    #[test]
    fn rejects_kernel_outputs_in_wrong_order() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(0), OutputId(0)),
            ],
        });

        let error = validate_kernel(&kernel).unwrap_err();
        assert_eq!(
            error.to_string(),
            "kernel output 0 must reference node 0 output 0"
        );
    }

    #[test]
    fn accepts_valid_kernel_graph() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: stage(1, vec![None]),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        });

        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: kernel,
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        assert!(validate_kernel_graph(&graph).is_ok());
    }

    #[test]
    fn rejects_kernel_graph_input_len_mismatch() {
        let kernel = Kernel(Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: stage(1, vec![None]),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        });

        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: kernel,
                inputs: vec![],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = validate_kernel_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: graph has 0 inputs for 1 kernel inputs"
        );
    }

    #[test]
    fn rejects_kernel_graph_output_len_mismatch() {
        let kernel = Kernel(Graph {
            inputs: vec![],
            nodes: vec![Node {
                inner: stage(0, vec![]),
                inputs: vec![],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        });

        let graph = Graph {
            inputs: vec![],
            nodes: vec![Node {
                inner: kernel,
                inputs: vec![],
                outputs: vec![],
            }],
            outputs: vec![],
        };

        let error = validate_kernel_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: graph has 0 outputs for 1 kernel outputs"
        );
    }
}

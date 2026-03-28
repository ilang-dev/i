use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::graph::validate_scheduled_stage_graph;
use crate::check::kernel::validate_kernel_graph;
use crate::ir::graph::{Graph, Input, InputId, Node, NodeId, Output, OutputId, Source};
use crate::ir::kernel::Kernel;
use crate::ir::stage::ScheduledStage;

pub fn lower_graph_to_kernel_graph(
    graph: &Graph<ScheduledStage>,
) -> Result<Graph<Kernel>, LowerError> {
    validate_scheduled_stage_graph(graph).map_err(LowerError::from_graph)?;
    reject_graph_input_compute_sites(graph)?;

    let materialized_roots = materialized_roots(graph);
    let mut root_kernel_map = BTreeMap::new();
    let mut kernels = Vec::new();

    for root in &materialized_roots {
        let built = build_kernel(graph, *root)?;
        let kernel_node_id = NodeId(kernels.len());
        let node_inputs = built
            .boundary_sources
            .iter()
            .map(|source| remap_boundary_source(*source, &root_kernel_map))
            .collect::<Result<Vec<_>, _>>()?;
        let node_outputs = vec![Output; built.kernel.0.outputs.len()];
        root_kernel_map.insert(*root, (kernel_node_id, built.root_output));
        kernels.push(Node {
            inner: built.kernel,
            inputs: node_inputs,
            outputs: node_outputs,
        });
    }

    let outputs = graph
        .outputs
        .iter()
        .copied()
        .map(|source| remap_graph_output(source, &root_kernel_map))
        .collect::<Result<Vec<_>, _>>()?;

    let kernel_graph = Graph {
        inputs: vec![Input; graph.inputs.len()],
        nodes: kernels,
        outputs,
    };
    validate_kernel_graph(&kernel_graph).map_err(LowerError::from_kernel_graph)?;
    Ok(kernel_graph)
}

fn reject_graph_input_compute_sites(graph: &Graph<ScheduledStage>) -> Result<(), LowerError> {
    for (node_index, node) in graph.nodes.iter().enumerate() {
        for (input_index, source) in node.inputs.iter().enumerate() {
            if matches!(source, Source::Input(_))
                && node.inner.schedule.compute_sites[input_index].is_some()
            {
                return Err(LowerError::new(format!(
                    "node {} input {} computes graph input inside the kernel",
                    node_index, input_index
                )));
            }
        }
    }

    Ok(())
}

fn materialized_roots(graph: &Graph<ScheduledStage>) -> BTreeSet<NodeId> {
    let mut roots = BTreeSet::new();

    for source in &graph.outputs {
        if let Source::Node(node, _) = source {
            roots.insert(*node);
        }
    }

    for node in &graph.nodes {
        for (input_index, source) in node.inputs.iter().enumerate() {
            if node.inner.schedule.compute_sites[input_index].is_none() {
                if let Source::Node(node, _) = source {
                    roots.insert(*node);
                }
            }
        }
    }

    roots
}

fn remap_boundary_source(
    source: Source,
    root_kernel_map: &BTreeMap<NodeId, (NodeId, OutputId)>,
) -> Result<Source, LowerError> {
    match source {
        Source::Input(input) => Ok(Source::Input(input)),
        Source::Node(node, OutputId(0)) => root_kernel_map
            .get(&node)
            .copied()
            .map(|(kernel, output)| Source::Node(kernel, output))
            .ok_or_else(|| {
                LowerError::new(format!(
                    "materialized producer node {} has no kernel",
                    node.0
                ))
            }),
        Source::Node(node, output) => Err(LowerError::new(format!(
            "materialized producer node {} references unsupported output {}",
            node.0, output.0
        ))),
    }
}

fn remap_graph_output(
    source: Source,
    root_kernel_map: &BTreeMap<NodeId, (NodeId, OutputId)>,
) -> Result<Source, LowerError> {
    match source {
        Source::Input(input) => Ok(Source::Input(input)),
        Source::Node(node, OutputId(0)) => root_kernel_map
            .get(&node)
            .copied()
            .map(|(kernel, output)| Source::Node(kernel, output))
            .ok_or_else(|| LowerError::new(format!("output node {} has no kernel", node.0))),
        Source::Node(node, output) => Err(LowerError::new(format!(
            "output node {} references unsupported output {}",
            node.0, output.0
        ))),
    }
}

fn build_kernel(graph: &Graph<ScheduledStage>, root: NodeId) -> Result<BuiltKernel, LowerError> {
    let mut builder = KernelBuilder {
        graph,
        boundary_inputs: Vec::new(),
        boundary_map: BTreeMap::new(),
        nodes: Vec::new(),
        outputs: Vec::new(),
    };
    let root_source = builder.build_instance(root)?;
    let root_output = match root_source {
        Source::Node(node, OutputId(0)) => OutputId(node.0),
        Source::Node(_, output) => output,
        Source::Input(_) => unreachable!("kernel root cannot be a graph input"),
    };

    Ok(BuiltKernel {
        kernel: Kernel(Graph {
            inputs: vec![Input; builder.boundary_inputs.len()],
            nodes: builder.nodes,
            outputs: builder.outputs,
        }),
        boundary_sources: builder.boundary_inputs,
        root_output,
    })
}

struct KernelBuilder<'a> {
    graph: &'a Graph<ScheduledStage>,
    boundary_inputs: Vec<Source>,
    boundary_map: BTreeMap<Source, InputId>,
    nodes: Vec<Node<ScheduledStage>>,
    outputs: Vec<Source>,
}

impl<'a> KernelBuilder<'a> {
    fn build_instance(&mut self, node_id: NodeId) -> Result<Source, LowerError> {
        let node = &self.graph.nodes[node_id.0];
        let mut inputs = Vec::with_capacity(node.inputs.len());

        for (input_index, source) in node.inputs.iter().copied().enumerate() {
            match (source, node.inner.schedule.compute_sites[input_index]) {
                (Source::Input(_), Some(_)) => {
                    return Err(LowerError::new(format!(
                        "node {} input {} computes graph input inside the kernel",
                        node_id.0, input_index
                    )))
                }
                (boundary, None) => {
                    let input = self.boundary_input(boundary);
                    inputs.push(Source::Input(input));
                }
                (Source::Node(child, OutputId(0)), Some(_)) => {
                    inputs.push(self.build_instance(child)?);
                }
                (Source::Node(child, output), Some(_)) => {
                    return Err(LowerError::new(format!(
                        "node {} input {} references unsupported producer output {} of node {}",
                        node_id.0, input_index, output.0, child.0
                    )))
                }
            }
        }

        let local_node = NodeId(self.nodes.len());
        self.nodes.push(Node {
            inner: node.inner.clone(),
            inputs,
            outputs: vec![Output],
        });
        let source = Source::Node(local_node, OutputId(0));
        self.outputs.push(source);
        Ok(source)
    }

    fn boundary_input(&mut self, source: Source) -> InputId {
        if let Some(input) = self.boundary_map.get(&source) {
            return *input;
        }

        let input = InputId(self.boundary_inputs.len());
        self.boundary_inputs.push(source);
        self.boundary_map.insert(source, input);
        input
    }
}

struct BuiltKernel {
    kernel: Kernel,
    boundary_sources: Vec<Source>,
    root_output: OutputId,
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

    fn from_graph(error: crate::check::graph::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_kernel_graph(error: crate::check::kernel::ValidationError) -> Self {
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
    use super::lower_graph_to_kernel_graph;
    use crate::ir::common::Op;
    use crate::ir::graph::{Graph, Input, InputId, Node, NodeId, Output, OutputId, Source};
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

    fn unary_input_stage() -> ScheduledStage {
        stage(1, vec![None])
    }

    #[test]
    fn lowers_single_output_stage_to_single_kernel() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: unary_input_stage(),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.inputs.len(), 1);
        assert_eq!(kernel_graph.nodes.len(), 1);
        assert_eq!(
            kernel_graph.nodes[0].inputs,
            vec![Source::Input(InputId(0))]
        );
        assert_eq!(kernel_graph.nodes[0].outputs.len(), 1);
        assert_eq!(
            kernel_graph.outputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );

        let kernel = &kernel_graph.nodes[0].inner.0;
        assert_eq!(kernel.inputs.len(), 1);
        assert_eq!(kernel.nodes.len(), 1);
        assert_eq!(kernel.outputs, vec![Source::Node(NodeId(0), OutputId(0))]);
    }

    #[test]
    fn lowers_none_edge_as_kernel_boundary() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(1), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 2);
        assert_eq!(
            kernel_graph.nodes[0].inputs,
            vec![Source::Input(InputId(0))]
        );
        assert_eq!(
            kernel_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(
            kernel_graph.outputs,
            vec![Source::Node(NodeId(1), OutputId(0))]
        );
    }

    #[test]
    fn lowers_some_edge_inside_consumer_kernel() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(1), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 1);
        assert_eq!(
            kernel_graph.nodes[0].inputs,
            vec![Source::Input(InputId(0))]
        );
        assert_eq!(kernel_graph.nodes[0].outputs.len(), 2);
        assert_eq!(
            kernel_graph.outputs,
            vec![Source::Node(NodeId(0), OutputId(1))]
        );

        let kernel = &kernel_graph.nodes[0].inner.0;
        assert_eq!(kernel.nodes.len(), 2);
        assert_eq!(
            kernel.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(
            kernel.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ]
        );
    }

    #[test]
    fn duplicates_per_compute_directive_inside_one_kernel() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(2, vec![Some(Site::Root), Some(Site::Root)]),
                    inputs: vec![
                        Source::Node(NodeId(1), OutputId(0)),
                        Source::Node(NodeId(2), OutputId(0)),
                    ],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(3), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();
        let kernel = &kernel_graph.nodes[0].inner.0;

        assert_eq!(kernel.nodes.len(), 5);
        assert_eq!(kernel.outputs.len(), 5);
        assert_eq!(kernel.nodes[0].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(kernel.nodes[2].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(
            kernel.nodes[4].inputs,
            vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(3), OutputId(0)),
            ]
        );
    }

    #[test]
    fn shares_one_materialized_kernel_for_multiple_none_consumers() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(2), OutputId(0)),
            ],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 3);
        assert_eq!(
            kernel_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(
            kernel_graph.nodes[2].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
    }

    #[test]
    fn duplicates_once_for_some_and_once_for_none() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(1), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(2), OutputId(0)),
                Source::Node(NodeId(3), OutputId(0)),
            ],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 3);
        assert_eq!(kernel_graph.nodes[2].outputs.len(), 3);
        assert_eq!(
            kernel_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );

        let some_kernel = &kernel_graph.nodes[2].inner.0;
        assert_eq!(some_kernel.nodes.len(), 3);
        assert_eq!(some_kernel.nodes[0].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(
            some_kernel.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(
            some_kernel.nodes[2].inputs,
            vec![Source::Node(NodeId(1), OutputId(0))]
        );
    }

    #[test]
    fn program_output_forces_materialized_kernel() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
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
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 2);
        assert_eq!(kernel_graph.nodes[0].outputs.len(), 1);
        assert_eq!(kernel_graph.nodes[1].outputs.len(), 2);
        assert_eq!(
            kernel_graph.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(1)),
            ]
        );
    }

    #[test]
    fn program_output_adds_materialized_copy_beside_two_compute_at_copies() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(2, vec![Some(Site::Root), Some(Site::Root)]),
                    inputs: vec![
                        Source::Node(NodeId(1), OutputId(0)),
                        Source::Node(NodeId(2), OutputId(0)),
                    ],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(3), OutputId(0)),
            ],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.nodes.len(), 2);
        assert_eq!(kernel_graph.nodes[0].outputs.len(), 1);
        assert_eq!(kernel_graph.nodes[1].outputs.len(), 5);
        assert_eq!(
            kernel_graph.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(4)),
            ]
        );
    }

    #[test]
    fn preserves_program_output_order() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![None]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(0), OutputId(0)),
            ],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(
            kernel_graph.outputs,
            vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(0), OutputId(0)),
            ]
        );
    }

    #[test]
    fn forwards_graph_input_outputs() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![],
            outputs: vec![Source::Input(InputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();

        assert_eq!(kernel_graph.inputs.len(), 1);
        assert!(kernel_graph.nodes.is_empty());
        assert_eq!(kernel_graph.outputs, vec![Source::Input(InputId(0))]);
    }

    #[test]
    fn rejects_compute_at_on_graph_input() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![Node {
                inner: stage(1, vec![Some(Site::Root)]),
                inputs: vec![Source::Input(InputId(0))],
                outputs: vec![Output],
            }],
            outputs: vec![Source::Node(NodeId(0), OutputId(0))],
        };

        let error = lower_graph_to_kernel_graph(&graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0 input 0 computes graph input inside the kernel"
        );
    }

    #[test]
    fn deduplicates_kernel_boundary_inputs() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(2, vec![None, None]),
                    inputs: vec![
                        Source::Node(NodeId(0), OutputId(0)),
                        Source::Node(NodeId(0), OutputId(0)),
                    ],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(1), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();
        let kernel = &kernel_graph.nodes[1].inner.0;

        assert_eq!(
            kernel_graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(kernel.inputs.len(), 1);
        assert_eq!(
            kernel.nodes[0].inputs,
            vec![Source::Input(InputId(0)), Source::Input(InputId(0))]
        );
    }

    #[test]
    fn records_branching_kernel_outputs_in_postorder() {
        let graph = Graph {
            inputs: vec![Input],
            nodes: vec![
                Node {
                    inner: unary_input_stage(),
                    inputs: vec![Source::Input(InputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(2, vec![Some(Site::Root), None]),
                    inputs: vec![
                        Source::Node(NodeId(1), OutputId(0)),
                        Source::Input(InputId(0)),
                    ],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(1, vec![Some(Site::Root)]),
                    inputs: vec![Source::Node(NodeId(0), OutputId(0))],
                    outputs: vec![Output],
                },
                Node {
                    inner: stage(2, vec![Some(Site::Root), Some(Site::Root)]),
                    inputs: vec![
                        Source::Node(NodeId(2), OutputId(0)),
                        Source::Node(NodeId(3), OutputId(0)),
                    ],
                    outputs: vec![Output],
                },
            ],
            outputs: vec![Source::Node(NodeId(4), OutputId(0))],
        };

        let kernel_graph = lower_graph_to_kernel_graph(&graph).unwrap();
        let kernel = &kernel_graph.nodes[0].inner.0;

        assert_eq!(
            kernel.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(2), OutputId(0)),
                Source::Node(NodeId(3), OutputId(0)),
                Source::Node(NodeId(4), OutputId(0)),
                Source::Node(NodeId(5), OutputId(0)),
            ]
        );
        assert_eq!(
            kernel_graph.outputs,
            vec![Source::Node(NodeId(0), OutputId(5))]
        );
    }
}

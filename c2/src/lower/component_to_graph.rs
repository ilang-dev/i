use std::fmt;

use crate::check::component::validate_component;
use crate::check::graph::validate_node_graph;
use crate::ir::component::Component;
use crate::ir::graph::{Graph, Input, InputId, Node, NodeId, Output, OutputId, Source};
use crate::ir::node::Node as IrNode;

use super::expr_to_node::{lower_expr_to_node_unchecked, LowerError as ExprLowerError};

pub fn lower_component_to_graph(component: &Component) -> Result<Graph<IrNode>, LowerError> {
    validate_component(component).map_err(LowerError::from_component)?;
    let partial = lower_component_to_partial(component)?;
    let graph = Graph {
        inputs: vec![Input; partial.inputs],
        nodes: partial.nodes,
        outputs: partial.outputs,
    };
    validate_node_graph(&graph).map_err(LowerError::from_graph)?;
    Ok(graph)
}

fn lower_component_to_partial(component: &Component) -> Result<PartialGraph, LowerError> {
    match component {
        Component::Expr(expr) => {
            let node = lower_expr_to_node_unchecked(expr).map_err(LowerError::from_expr)?;
            Ok(PartialGraph {
                inputs: node.inputs.len(),
                nodes: vec![Node {
                    inner: node,
                    inputs: (0..expr.inputs.len())
                        .map(|index| Source::Input(InputId(index)))
                        .collect(),
                    outputs: vec![Output],
                }],
                outputs: vec![Source::Node(NodeId(0), OutputId(0))],
            })
        }
        Component::Compose(left, right) => {
            let left = lower_component_to_partial(left)?;
            let right = lower_component_to_partial(right)?;
            Ok(compose_graphs(left, right))
        }
        Component::Chain(left, right) => {
            let left = lower_component_to_partial(left)?;
            let right = lower_component_to_partial(right)?;
            Ok(chain_graphs(left, right))
        }
        Component::Fanout(left, right) => {
            let left = lower_component_to_partial(left)?;
            let right = lower_component_to_partial(right)?;
            Ok(fanout_graphs(left, right))
        }
        Component::Pair(left, right) => {
            let left = lower_component_to_partial(left)?;
            let right = lower_component_to_partial(right)?;
            Ok(pair_graphs(left, right))
        }
        Component::Swap(inner) => {
            let mut inner = lower_component_to_partial(inner)?;
            if inner.outputs.len() >= 2 {
                inner.outputs.swap(0, 1);
            }
            Ok(inner)
        }
    }
}

fn pair_graphs(left: PartialGraph, right: PartialGraph) -> PartialGraph {
    let left_input_map = input_sources(0, left.inputs);
    let left_outputs = remap_outputs(&left.outputs, &left_input_map, 0);
    let left_nodes = remap_nodes(left.nodes, &left_input_map, 0);

    let right_input_base = left.inputs;
    let right_input_map = input_sources(right_input_base, right.inputs);
    let right_node_offset = left_nodes.len();
    let right_outputs = remap_outputs(&right.outputs, &right_input_map, right_node_offset);
    let right_nodes = remap_nodes(right.nodes, &right_input_map, right_node_offset);

    PartialGraph {
        inputs: right_input_base + right.inputs,
        nodes: left_nodes.into_iter().chain(right_nodes).collect(),
        outputs: left_outputs.into_iter().chain(right_outputs).collect(),
    }
}

fn chain_graphs(left: PartialGraph, right: PartialGraph) -> PartialGraph {
    let paired = left.outputs.len().min(right.inputs);

    let left_input_map = input_sources(0, left.inputs);
    let left_outputs = remap_outputs(&left.outputs, &left_input_map, 0);
    let left_nodes = remap_nodes(left.nodes, &left_input_map, 0);

    let mut right_input_map = left_outputs[..paired].to_vec();
    right_input_map.extend(
        (0..(right.inputs - paired)).map(|index| Source::Input(InputId(left.inputs + index))),
    );
    let right_node_offset = left_nodes.len();
    let right_outputs = remap_outputs(&right.outputs, &right_input_map, right_node_offset);
    let right_nodes = remap_nodes(right.nodes, &right_input_map, right_node_offset);

    PartialGraph {
        inputs: left.inputs + right.inputs - paired,
        nodes: left_nodes.into_iter().chain(right_nodes).collect(),
        outputs: left_outputs[paired..]
            .iter()
            .copied()
            .chain(right_outputs)
            .collect(),
    }
}

fn compose_graphs(left: PartialGraph, right: PartialGraph) -> PartialGraph {
    let paired = left.inputs.min(right.outputs.len());

    let right_input_map = input_sources(0, right.inputs);
    let right_outputs = remap_outputs(&right.outputs, &right_input_map, 0);
    let right_nodes = remap_nodes(right.nodes, &right_input_map, 0);

    let mut left_input_map = right_outputs[..paired].to_vec();
    left_input_map.extend(
        (0..(left.inputs - paired)).map(|index| Source::Input(InputId(right.inputs + index))),
    );
    let left_node_offset = right_nodes.len();
    let left_outputs = remap_outputs(&left.outputs, &left_input_map, left_node_offset);
    let left_nodes = remap_nodes(left.nodes, &left_input_map, left_node_offset);

    PartialGraph {
        inputs: right.inputs + left.inputs - paired,
        nodes: right_nodes.into_iter().chain(left_nodes).collect(),
        outputs: left_outputs
            .into_iter()
            .chain(right_outputs[paired..].iter().copied())
            .collect(),
    }
}

fn fanout_graphs(left: PartialGraph, right: PartialGraph) -> PartialGraph {
    let paired = left.inputs.min(right.inputs);

    let left_input_map = input_sources(0, left.inputs);
    let left_outputs = remap_outputs(&left.outputs, &left_input_map, 0);
    let left_nodes = remap_nodes(left.nodes, &left_input_map, 0);

    let mut right_input_map = input_sources(0, paired);
    right_input_map.extend(
        (0..(right.inputs - paired)).map(|index| Source::Input(InputId(left.inputs + index))),
    );
    let right_node_offset = left_nodes.len();
    let right_outputs = remap_outputs(&right.outputs, &right_input_map, right_node_offset);
    let right_nodes = remap_nodes(right.nodes, &right_input_map, right_node_offset);

    PartialGraph {
        inputs: left.inputs + right.inputs - paired,
        nodes: left_nodes.into_iter().chain(right_nodes).collect(),
        outputs: left_outputs.into_iter().chain(right_outputs).collect(),
    }
}

fn input_sources(base: usize, len: usize) -> Vec<Source> {
    (0..len)
        .map(|index| Source::Input(InputId(base + index)))
        .collect()
}

fn remap_nodes(
    nodes: Vec<Node<IrNode>>,
    input_map: &[Source],
    node_offset: usize,
) -> Vec<Node<IrNode>> {
    nodes
        .into_iter()
        .map(|node| Node {
            inner: node.inner,
            inputs: node
                .inputs
                .into_iter()
                .map(|source| remap_source(source, input_map, node_offset))
                .collect(),
            outputs: node.outputs,
        })
        .collect()
}

fn remap_outputs(outputs: &[Source], input_map: &[Source], node_offset: usize) -> Vec<Source> {
    outputs
        .iter()
        .copied()
        .map(|source| remap_source(source, input_map, node_offset))
        .collect()
}

fn remap_source(source: Source, input_map: &[Source], node_offset: usize) -> Source {
    match source {
        Source::Input(InputId(index)) => input_map[index],
        Source::Node(NodeId(index), output) => Source::Node(NodeId(node_offset + index), output),
    }
}

struct PartialGraph {
    inputs: usize,
    nodes: Vec<Node<IrNode>>,
    outputs: Vec<Source>,
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

    fn from_component(error: crate::check::component::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_expr(error: ExprLowerError) -> Self {
        Self::new(error.to_string())
    }

    fn from_graph(error: crate::check::graph::ValidationError) -> Self {
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
    use crate::component;
    use crate::front::parse_expr;
    use crate::ir::common::{Index, Op};
    use crate::ir::graph::{InputId, NodeId, OutputId, Source};
    use crate::ir::node::{AxisRef, Site, SplitList};

    use super::lower_component_to_graph;

    fn parse_component_expr(src: &str) -> crate::ir::component::Component {
        component::expr(parse_expr(src).unwrap())
    }

    #[test]
    fn lowers_single_expr_component() {
        let graph = lower_component_to_graph(&parse_component_expr("ij+i~ij")).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.outputs, vec![Source::Node(NodeId(0), OutputId(0))]);
        assert_eq!(
            graph.nodes[0].inputs,
            vec![Source::Input(InputId(0)), Source::Input(InputId(1))]
        );
        assert_eq!(graph.nodes[0].inner.op, Op::Add);
    }

    #[test]
    fn lowers_pair_by_concatenating_boundaries() {
        let component = parse_component_expr("ij+i~ij").pair(parse_component_expr("+ij~i"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 3);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(
            graph.nodes[0].inputs,
            vec![Source::Input(InputId(0)), Source::Input(InputId(1))]
        );
        assert_eq!(graph.nodes[1].inputs, vec![Source::Input(InputId(2))]);
        assert_eq!(
            graph.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ]
        );
    }

    #[test]
    fn lowers_chain_by_wiring_left_outputs_to_right_inputs() {
        let component = parse_component_expr("ik*kj~ijk").chain(parse_component_expr("+ijk~ij"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(
            graph.nodes[0].inputs,
            vec![Source::Input(InputId(0)), Source::Input(InputId(1))]
        );
        assert_eq!(
            graph.nodes[1].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
        assert_eq!(graph.outputs, vec![Source::Node(NodeId(1), OutputId(0))]);
        assert_eq!(
            graph.nodes[1].inner.init_site,
            Some(Site::At(AxisRef {
                index: Index(1),
                level: 0
            }))
        );
    }

    #[test]
    fn lowers_chain_mixed_arity_with_leftover_boundaries() {
        let component = parse_component_expr("+ij~i")
            .pair(parse_component_expr("+ij~j"))
            .chain(parse_component_expr("i~i"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(
            graph.outputs,
            vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(2), OutputId(0)),
            ]
        );
        assert_eq!(
            graph.nodes[2].inputs,
            vec![Source::Node(NodeId(0), OutputId(0))]
        );
    }

    #[test]
    fn lowers_compose_by_wiring_right_outputs_to_left_inputs() {
        let component = parse_component_expr("ij+i~ij").compose(parse_component_expr("+ij~i"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(
            graph.nodes[1].inputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Input(InputId(1)),
            ]
        );
        assert_eq!(graph.outputs, vec![Source::Node(NodeId(1), OutputId(0))]);
    }

    #[test]
    fn lowers_compose_mixed_arity_with_leftover_boundaries() {
        let component = parse_component_expr("i+i~i")
            .compose(parse_component_expr("i~i").pair(parse_component_expr("i~i")));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.outputs, vec![Source::Node(NodeId(2), OutputId(0))]);
        assert_eq!(
            graph.nodes[2].inputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ]
        );
    }

    #[test]
    fn lowers_fanout_by_sharing_inputs_pairwise() {
        let component = parse_component_expr("+ij~i").fanout(parse_component_expr("+ij~j"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(graph.nodes[1].inputs, vec![Source::Input(InputId(0))]);
        assert_eq!(
            graph.outputs,
            vec![
                Source::Node(NodeId(0), OutputId(0)),
                Source::Node(NodeId(1), OutputId(0)),
            ]
        );
    }

    #[test]
    fn lowers_fanout_mixed_arity_with_unpaired_inputs() {
        let component = parse_component_expr("ij+i~ij").fanout(parse_component_expr("+ij~i"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(
            graph.nodes[0].inputs,
            vec![Source::Input(InputId(0)), Source::Input(InputId(1))]
        );
        assert_eq!(graph.nodes[1].inputs, vec![Source::Input(InputId(0))]);
    }

    #[test]
    fn lowers_swap_by_swapping_first_two_outputs() {
        let component = parse_component_expr("+ij~i")
            .pair(parse_component_expr("+ij~j"))
            .swap();
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(
            graph.outputs,
            vec![
                Source::Node(NodeId(1), OutputId(0)),
                Source::Node(NodeId(0), OutputId(0)),
            ]
        );
    }

    #[test]
    fn leaves_single_output_swap_unchanged() {
        let component = parse_component_expr("+ij~i").swap();
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.outputs, vec![Source::Node(NodeId(0), OutputId(0))]);
    }

    #[test]
    fn lowers_nested_component_deterministically() {
        let component = parse_component_expr("ik*kj~ijk")
            .fanout(parse_component_expr("+ik~i"))
            .chain(parse_component_expr("+ijk~ij").pair(parse_component_expr("i~i")))
            .swap();
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(
            graph.outputs,
            vec![
                Source::Node(NodeId(3), OutputId(0)),
                Source::Node(NodeId(2), OutputId(0)),
            ]
        );
    }

    #[test]
    fn propagates_component_validation_errors() {
        let component = parse_component_expr("ii~ii");
        let error = lower_component_to_graph(&component).unwrap_err();

        assert_eq!(error.to_string(), "expr 0: output pattern repeats axis `i`");
    }

    #[test]
    fn produced_graph_passes_validation() {
        let component = parse_component_expr("ik*kj~ijk")
            .chain(parse_component_expr("+ijk~ij"))
            .fanout(parse_component_expr("ij~ij"));
        let graph = lower_component_to_graph(&component).unwrap();

        assert_eq!(
            graph.nodes[2].inner.splits,
            vec![SplitList(vec![]), SplitList(vec![])]
        );
    }
}

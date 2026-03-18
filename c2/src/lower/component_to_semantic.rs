use std::collections::{BTreeMap, BTreeSet};

use crate::ir::common::{ExprId, Extent, Pattern, Scalar, Shape, TensorType, ValueId};
use crate::ir::component::{Component, Expr};
use crate::ir::semantic_graph::{Axis, Graph, Index, LocalExtentSource, Stage, Use};

pub fn lower_component_to_semantic(component: &Component) -> Graph {
    let mut next_input_key = 0usize;
    let fragment = lower_component(component, &mut next_input_key);
    build_graph(fragment)
}

#[derive(Clone, Debug)]
struct Fragment {
    inputs: Vec<InputKey>,
    outputs: Vec<Source>,
    input_ranks: BTreeMap<InputKey, usize>,
    stages: BTreeMap<ExprId, StageNode>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct InputKey(usize);

#[derive(Clone, Debug, Eq, PartialEq)]
enum Source {
    Input(InputKey),
    Stage(ExprId),
}

#[derive(Clone, Debug)]
struct StageNode {
    expr: Expr,
    inputs: Vec<Source>,
}

fn lower_component(component: &Component, next_input_key: &mut usize) -> Fragment {
    match component {
        Component::Expr(expr) => lower_expr(expr, next_input_key),
        Component::Compose(left, right) => compose_fragments(
            lower_component(left, next_input_key),
            lower_component(right, next_input_key),
        ),
        Component::Chain(left, right) => compose_fragments(
            lower_component(right, next_input_key),
            lower_component(left, next_input_key),
        ),
        Component::Fanout(left, right) => fanout_fragments(
            lower_component(left, next_input_key),
            lower_component(right, next_input_key),
        ),
        Component::Pair(left, right) => pair_fragments(
            lower_component(left, next_input_key),
            lower_component(right, next_input_key),
        ),
        Component::Swap(inner) => swap_fragment(lower_component(inner, next_input_key)),
    }
}

fn lower_expr(expr: &Expr, next_input_key: &mut usize) -> Fragment {
    let mut inputs = Vec::with_capacity(expr.inputs.len());
    let mut input_ranks = BTreeMap::new();

    for pattern in &expr.inputs {
        let key = InputKey(*next_input_key);
        *next_input_key += 1;
        input_ranks.insert(key, pattern.0.len());
        inputs.push(key);
    }

    let mut stages = BTreeMap::new();
    stages.insert(
        expr.id,
        StageNode {
            expr: expr.clone(),
            inputs: inputs.iter().copied().map(Source::Input).collect(),
        },
    );

    Fragment {
        inputs,
        outputs: vec![Source::Stage(expr.id)],
        input_ranks,
        stages,
    }
}

fn compose_fragments(mut left: Fragment, right: Fragment) -> Fragment {
    let consumed = left.inputs.len().min(right.outputs.len());
    let rewrites = left
        .inputs
        .iter()
        .take(consumed)
        .copied()
        .zip(right.outputs.iter().take(consumed).cloned())
        .collect::<Vec<_>>();

    for (input, replacement) in rewrites {
        rewrite_input_source(&mut left, input, replacement);
    }

    let mut inputs = right.inputs;
    inputs.extend(left.inputs.into_iter().skip(consumed));

    let mut outputs = left.outputs;
    outputs.extend(right.outputs.into_iter().skip(consumed));

    Fragment {
        inputs,
        outputs,
        input_ranks: merge_input_ranks(left.input_ranks, right.input_ranks),
        stages: merge_stages(right.stages, left.stages),
    }
}

fn fanout_fragments(left: Fragment, mut right: Fragment) -> Fragment {
    let consumed = left.inputs.len().min(right.inputs.len());
    let rewrites = left
        .inputs
        .iter()
        .take(consumed)
        .copied()
        .zip(right.inputs.iter().take(consumed).copied())
        .collect::<Vec<_>>();

    for (left_input, right_input) in rewrites {
        rewrite_input_source(&mut right, right_input, Source::Input(left_input));
    }

    let mut inputs = left.inputs;
    inputs.extend(right.inputs.into_iter().skip(consumed));

    let mut outputs = left.outputs;
    outputs.extend(right.outputs);

    Fragment {
        inputs,
        outputs,
        input_ranks: merge_input_ranks(left.input_ranks, right.input_ranks),
        stages: merge_stages(left.stages, right.stages),
    }
}

fn pair_fragments(left: Fragment, right: Fragment) -> Fragment {
    let mut inputs = left.inputs;
    inputs.extend(right.inputs);

    let mut outputs = left.outputs;
    outputs.extend(right.outputs);

    Fragment {
        inputs,
        outputs,
        input_ranks: merge_input_ranks(left.input_ranks, right.input_ranks),
        stages: merge_stages(left.stages, right.stages),
    }
}

fn swap_fragment(mut fragment: Fragment) -> Fragment {
    if fragment.outputs.len() >= 2 {
        fragment.outputs.swap(0, 1);
    }
    fragment
}

fn rewrite_input_source(fragment: &mut Fragment, input: InputKey, replacement: Source) {
    for stage in fragment.stages.values_mut() {
        for source in &mut stage.inputs {
            if matches!(source, Source::Input(candidate) if *candidate == input) {
                *source = replacement.clone();
            }
        }
    }

    for source in &mut fragment.outputs {
        if matches!(source, Source::Input(candidate) if *candidate == input) {
            *source = replacement.clone();
        }
    }
}

fn merge_input_ranks(
    mut left: BTreeMap<InputKey, usize>,
    right: BTreeMap<InputKey, usize>,
) -> BTreeMap<InputKey, usize> {
    for (key, rank) in right {
        let prev = left.insert(key, rank);
        assert!(prev.is_none(), "duplicate input key");
    }
    left
}

fn merge_stages(
    mut left: BTreeMap<ExprId, StageNode>,
    right: BTreeMap<ExprId, StageNode>,
) -> BTreeMap<ExprId, StageNode> {
    for (expr, stage) in right {
        let prev = left.insert(expr, stage);
        assert!(prev.is_none(), "duplicate expr id {expr}");
    }
    left
}

fn build_graph(fragment: Fragment) -> Graph {
    let stage_order = topo_order(&fragment);

    let mut input_values = BTreeMap::new();
    let mut input_extents = BTreeMap::new();
    let inputs = fragment
        .inputs
        .iter()
        .enumerate()
        .map(|(input_index, input)| {
            input_values.insert(*input, input_index);
            let rank = fragment.input_ranks[input];
            let dims = (0..rank)
                .map(|dim| Extent::Param(format!("input{input_index}_dim{dim}")))
                .collect::<Vec<_>>();
            input_extents.insert(*input, dims.clone());
            TensorType {
                scalar: Scalar::Float,
                shape: Shape(dims),
            }
        })
        .collect();

    let mut stage_values = BTreeMap::new();
    for (offset, expr) in stage_order.iter().enumerate() {
        stage_values.insert(*expr, fragment.inputs.len() + offset);
    }

    let stages = stage_order
        .iter()
        .map(|expr| {
            let node = &fragment.stages[expr];
            let value = stage_values[expr];
            let axis_order = collect_stage_axes(&node.expr);
            let axis_positions = axis_order
                .iter()
                .enumerate()
                .map(|(index, axis)| (*axis, index))
                .collect::<BTreeMap<_, _>>();

            let axis_extents = axis_order
                .iter()
                .map(|axis| resolve_local_axis_extent(*axis, &node.expr))
                .collect::<Vec<_>>();

            let inputs = node
                .inputs
                .iter()
                .zip(&node.expr.inputs)
                .map(|(source, pattern)| Use {
                    value: resolve_value_id(source, &input_values, &stage_values),
                    index: pattern_to_index(pattern, &axis_positions),
                })
                .collect::<Vec<_>>();

            let output = pattern_to_index(&node.expr.output, &axis_positions);

            Stage {
                value,
                expr: node.expr.id,
                op: node.expr.op,
                inputs,
                axes: axis_extents
                    .into_iter()
                    .map(|extent| Axis { extent })
                    .collect(),
                output,
            }
        })
        .collect();

    let outputs = fragment
        .outputs
        .iter()
        .map(|source| resolve_value_id(source, &input_values, &stage_values))
        .collect();

    Graph {
        inputs,
        stages,
        outputs,
    }
}

fn topo_order(fragment: &Fragment) -> Vec<ExprId> {
    let mut indegree = fragment
        .stages
        .keys()
        .copied()
        .map(|expr| (expr, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents: BTreeMap<ExprId, Vec<ExprId>> = BTreeMap::new();

    for (expr, node) in &fragment.stages {
        let deps = node
            .inputs
            .iter()
            .filter_map(|source| match source {
                Source::Input(_) => None,
                Source::Stage(dep) => Some(*dep),
            })
            .collect::<BTreeSet<_>>();

        indegree.insert(*expr, deps.len());
        for dep in deps {
            dependents.entry(dep).or_default().push(*expr);
        }
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(expr, degree)| (*degree == 0).then_some(*expr))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(fragment.stages.len());

    while let Some(expr) = ready.first().copied() {
        ready.remove(&expr);
        order.push(expr);

        for dependent in dependents.get(&expr).into_iter().flatten() {
            let degree = indegree
                .get_mut(dependent)
                .expect("dependent stage must exist");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(*dependent);
            }
        }
    }

    assert_eq!(
        order.len(),
        fragment.stages.len(),
        "component graph must be acyclic"
    );
    order
}

fn collect_stage_axes(expr: &Expr) -> Vec<char> {
    let mut axes = Vec::new();
    let mut seen = BTreeSet::new();

    for axis in &expr.output.0 {
        if seen.insert(*axis) {
            axes.push(*axis);
        }
    }

    for pattern in &expr.inputs {
        for axis in &pattern.0 {
            if seen.insert(*axis) {
                axes.push(*axis);
            }
        }
    }

    axes
}

fn resolve_local_axis_extent(axis: char, expr: &Expr) -> LocalExtentSource {
    for (input_index, pattern) in expr.inputs.iter().enumerate() {
        for (dim, pattern_axis) in pattern.0.iter().enumerate() {
            if *pattern_axis != axis {
                continue;
            }

            return LocalExtentSource::InputDim {
                input: input_index,
                dim,
            };
        }
    }

    panic!(
        "axis `{axis}` must appear in one of expr {} input patterns",
        expr.id
    );
}

fn pattern_to_index(pattern: &Pattern, axis_positions: &BTreeMap<char, usize>) -> Index {
    Index(
        pattern
            .0
            .iter()
            .map(|axis| axis_positions[axis])
            .collect::<Vec<_>>(),
    )
}

fn resolve_value_id(
    source: &Source,
    input_values: &BTreeMap<InputKey, ValueId>,
    stage_values: &BTreeMap<ExprId, ValueId>,
) -> ValueId {
    match source {
        Source::Input(input) => input_values[input],
        Source::Stage(expr) => stage_values[expr],
    }
}

#[cfg(test)]
mod tests {
    use super::lower_component_to_semantic;
    use crate::component;
    use crate::ir::common::{Extent, Op, Pattern, Scalar, Shape, TensorType};
    use crate::ir::component::{Expr, Schedule};
    use crate::ir::semantic_graph::{Axis, Graph, Index, LocalExtentSource, Stage, Use};

    #[test]
    fn lowers_pointwise_broadcast_to_explicit_indices() {
        let graph = lower_component_to_semantic(&component::expr(make_expr(
            7,
            Op::Add,
            &["ij", "i"],
            "ij",
        )));

        assert_eq!(
            graph,
            Graph {
                inputs: vec![float_input(0, 2), float_input(1, 1)],
                stages: vec![Stage {
                    value: 2,
                    expr: 7,
                    op: Op::Add,
                    inputs: vec![
                        Use {
                            value: 0,
                            index: Index(vec![0, 1]),
                        },
                        Use {
                            value: 1,
                            index: Index(vec![0]),
                        },
                    ],
                    axes: vec![axis(local_input_dim(0, 0)), axis(local_input_dim(0, 1))],
                    output: Index(vec![0, 1]),
                }],
                outputs: vec![2],
            }
        );
    }

    #[test]
    fn lowers_reduction_with_output_axes_first() {
        let graph =
            lower_component_to_semantic(&component::expr(make_expr(11, Op::Add, &["ijk"], "ij")));

        assert_eq!(
            graph,
            Graph {
                inputs: vec![float_input(0, 3)],
                stages: vec![Stage {
                    value: 1,
                    expr: 11,
                    op: Op::Add,
                    inputs: vec![Use {
                        value: 0,
                        index: Index(vec![0, 1, 2]),
                    }],
                    axes: vec![
                        axis(local_input_dim(0, 0)),
                        axis(local_input_dim(0, 1)),
                        axis(local_input_dim(0, 2)),
                    ],
                    output: Index(vec![0, 1]),
                }],
                outputs: vec![1],
            }
        );
    }

    #[test]
    fn lowers_cross_input_indices_in_stage_local_axis_order() {
        let graph = lower_component_to_semantic(&component::expr(make_expr(
            5,
            Op::Mul,
            &["ik", "kj"],
            "ijk",
        )));

        assert_eq!(
            graph,
            Graph {
                inputs: vec![float_input(0, 2), float_input(1, 2)],
                stages: vec![Stage {
                    value: 2,
                    expr: 5,
                    op: Op::Mul,
                    inputs: vec![
                        Use {
                            value: 0,
                            index: Index(vec![0, 2]),
                        },
                        Use {
                            value: 1,
                            index: Index(vec![2, 1]),
                        },
                    ],
                    axes: vec![
                        axis(local_input_dim(0, 0)),
                        axis(local_input_dim(1, 1)),
                        axis(local_input_dim(0, 1)),
                    ],
                    output: Index(vec![0, 1, 2]),
                }],
                outputs: vec![2],
            }
        );
    }

    #[test]
    fn lowers_fanout_then_chain_as_one_dataflow_graph() {
        let id = component::expr(make_expr(99, Op::Add, &["ij"], "ij"));
        let row_sum = component::expr(make_expr(99, Op::Add, &["ij"], "i"));
        let row_div = component::expr(make_expr(99, Op::Div, &["ij", "i"], "ij"));
        let component = id.fanout(row_sum).chain(row_div).finalize();

        let graph = lower_component_to_semantic(&component);

        assert_eq!(graph.inputs, vec![float_input(0, 2)]);
        assert_eq!(graph.outputs, vec![3]);
        assert_eq!(
            graph
                .stages
                .iter()
                .map(|stage| stage.expr)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            graph.stages[2].inputs,
            vec![
                Use {
                    value: 1,
                    index: Index(vec![0, 1]),
                },
                Use {
                    value: 2,
                    index: Index(vec![0]),
                },
            ]
        );
    }

    #[test]
    fn lowers_compose_with_leftover_roots_and_topological_value_ids() {
        let left = component::expr(make_expr(99, Op::Add, &["i"], "i"))
            .pair(component::expr(make_expr(99, Op::Add, &["j"], "j")));
        let right = component::expr(make_expr(99, Op::Add, &["i"], "i"))
            .pair(component::expr(make_expr(99, Op::Add, &["j"], "j")))
            .pair(component::expr(make_expr(99, Op::Add, &["k"], "k")));
        let component = left.compose(right).finalize();

        let graph = lower_component_to_semantic(&component);

        assert_eq!(
            graph.inputs,
            vec![float_input(0, 1), float_input(1, 1), float_input(2, 1)]
        );
        assert_eq!(
            graph
                .stages
                .iter()
                .map(|stage| stage.expr)
                .collect::<Vec<_>>(),
            vec![2, 0, 3, 1, 4]
        );
        assert_eq!(graph.outputs, vec![4, 6, 7]);
        assert_eq!(
            graph.stages[1].inputs,
            vec![Use {
                value: 3,
                index: Index(vec![0]),
            }]
        );
        assert_eq!(
            graph.stages[3].inputs,
            vec![Use {
                value: 5,
                index: Index(vec![0]),
            }]
        );
    }

    #[test]
    fn lowers_swap_by_reordering_graph_outputs() {
        let component = component::expr(make_expr(99, Op::Add, &["i"], "i"))
            .pair(component::expr(make_expr(99, Op::Add, &["j"], "j")))
            .swap()
            .finalize();

        let graph = lower_component_to_semantic(&component);

        assert_eq!(
            graph
                .stages
                .iter()
                .map(|stage| stage.expr)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(graph.outputs, vec![3, 2]);
    }

    fn make_expr(id: usize, op: Op, inputs: &[&str], output: &str) -> Expr {
        Expr {
            id,
            op,
            inputs: inputs
                .iter()
                .map(|pattern| Pattern(pattern.chars().collect()))
                .collect(),
            output: Pattern(output.chars().collect()),
            schedule: Schedule::default(),
        }
    }

    fn float_input(index: usize, rank: usize) -> TensorType {
        TensorType {
            scalar: Scalar::Float,
            shape: Shape(
                (0..rank)
                    .map(|dim| param(&format!("input{index}_dim{dim}")))
                    .collect(),
            ),
        }
    }

    fn local_input_dim(input: usize, dim: usize) -> LocalExtentSource {
        LocalExtentSource::InputDim { input, dim }
    }

    fn axis(extent: LocalExtentSource) -> Axis {
        Axis { extent }
    }

    fn param(name: &str) -> Extent {
        Extent::Param(name.to_string())
    }
}

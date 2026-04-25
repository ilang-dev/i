use std::collections::BTreeMap;
use std::fmt;

use crate::check::component::validate_component;
use crate::check::node::{required_init_site_from_order, validate_node};
use crate::ir::common::Index;
use crate::ir::component::Component;
use crate::ir::expr::{Expr, PermutationAtom};
use crate::ir::node::{AxisRef, MultiIndex, Node, Site, SplitFactor, SplitList};

pub fn lower_expr_to_node(expr: &Expr) -> Result<Node, LowerError> {
    validate_component(&Component::Expr(expr.clone())).map_err(LowerError::from_component)?;
    lower_expr_to_node_unchecked(expr)
}

pub(crate) fn lower_expr_to_node_unchecked(expr: &Expr) -> Result<Node, LowerError> {
    let axis_order = canonical_axis_order(expr);
    let axis_map = axis_order
        .iter()
        .enumerate()
        .map(|(index, axis)| (*axis, Index(index)))
        .collect::<BTreeMap<_, _>>();

    let mut node = Node {
        op: expr.op,
        rank: axis_order.len(),
        inputs: expr
            .inputs
            .iter()
            .map(|pattern| lower_multi_index(pattern, &axis_map))
            .collect::<Result<_, _>>()?,
        output: lower_multi_index(&expr.output, &axis_map)?,
        splits: lower_splits(expr, &axis_order),
        order: Vec::new(),
        compute_sites: Vec::new(),
        init_site: None,
    };

    let (order, compute_sites, init_site) = if expr.permutation.is_empty() {
        let order = default_order(node.splits.as_slice());
        let init_site =
            required_init_site_from_order(node.rank, &node.output, node.splits.as_slice(), &order)
                .map_err(LowerError::from_node)?;
        (order, vec![None; node.inputs.len()], init_site)
    } else {
        lower_schedule_from_permutation(expr, &node, &axis_map)?
    };

    node.order = order;
    node.compute_sites = compute_sites;
    node.init_site = init_site;
    validate_node(&node).map_err(LowerError::from_node)?;
    Ok(node)
}

fn canonical_axis_order(expr: &Expr) -> Vec<char> {
    let mut axes = Vec::new();

    for axis in &expr.output {
        if !axes.contains(axis) {
            axes.push(*axis);
        }
    }

    for input in &expr.inputs {
        for axis in input {
            if !axes.contains(axis) {
                axes.push(*axis);
            }
        }
    }

    axes
}

fn lower_multi_index(
    pattern: &[char],
    axis_map: &BTreeMap<char, Index>,
) -> Result<MultiIndex, LowerError> {
    pattern
        .iter()
        .map(|axis| {
            axis_map
                .get(axis)
                .copied()
                .ok_or_else(|| LowerError::new(format!("unknown local axis `{axis}`")))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(MultiIndex)
}

fn lower_splits(expr: &Expr, axis_order: &[char]) -> Vec<SplitList> {
    let split_map = expr
        .splits
        .iter()
        .map(|(axis, factors)| (*axis, factors))
        .collect::<BTreeMap<_, _>>();

    axis_order
        .iter()
        .map(|axis| {
            SplitList(
                split_map
                    .get(axis)
                    .into_iter()
                    .flat_map(|factors| factors.iter())
                    .map(|factor| SplitFactor(*factor))
                    .collect(),
            )
        })
        .collect()
}

fn default_order(splits: &[SplitList]) -> Vec<AxisRef> {
    let mut order = Vec::new();
    for (index, split_list) in splits.iter().enumerate() {
        for level in 0..=split_list.0.len() {
            order.push(AxisRef {
                index: Index(index),
                level,
            });
        }
    }
    order
}

fn lower_schedule_from_permutation(
    expr: &Expr,
    node: &Node,
    axis_map: &BTreeMap<char, Index>,
) -> Result<(Vec<AxisRef>, Vec<Option<Site>>, Option<Site>), LowerError> {
    let mut order = Vec::new();
    let mut compute_sites = vec![None; expr.inputs.len()];
    let mut last_axis_ref = None;
    let mut explicit_init_site = None;

    for atom in &expr.permutation {
        match atom {
            PermutationAtom::Axis { axis, part } => {
                let index = axis_map
                    .get(axis)
                    .copied()
                    .ok_or_else(|| LowerError::new(format!("unknown local axis `{axis}`")))?;
                let axis_ref = AxisRef {
                    index,
                    level: *part,
                };
                order.push(axis_ref);
                last_axis_ref = Some(axis_ref);
            }
            PermutationAtom::Input(input) => {
                let axis_ref = last_axis_ref.ok_or_else(|| {
                    LowerError::new(format!(
                        "input {} cannot be computed before any loop",
                        input
                    ))
                })?;
                let site = Site::At(axis_ref);
                let slot = compute_sites.get_mut(*input).ok_or_else(|| {
                    LowerError::new(format!(
                        "permutation references nonexistent input {}",
                        input
                    ))
                })?;
                *slot = Some(site);
            }
            PermutationAtom::Bang => {
                explicit_init_site = Some(last_axis_ref.map(Site::At).unwrap_or(Site::Root));
            }
        }
    }

    let required_init_site =
        required_init_site_from_order(node.rank, &node.output, node.splits.as_slice(), &order)
            .map_err(LowerError::from_node)?;
    let init_site = match (required_init_site, explicit_init_site) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(LowerError::new(
                "pointwise node cannot have an output init directive",
            ))
        }
        (Some(site), None) => Some(site),
        (Some(expected), Some(actual)) if expected == actual => Some(actual),
        (Some(expected), Some(_)) => {
            return Err(LowerError::new(format!(
                "output init directive must appear at {}",
                format_site(expected)
            )))
        }
    };

    Ok((order, compute_sites, init_site))
}

fn format_site(site: Site) -> String {
    match site {
        Site::Root => "Root".to_string(),
        Site::At(axis_ref) => format!("At({}{})", axis_ref.index.0, "'".repeat(axis_ref.level)),
    }
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

    fn from_node(error: crate::check::node::ValidationError) -> Self {
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
    use super::lower_expr_to_node;
    use crate::front::parse_expr;
    use crate::ir::common::{Index, Op};
    use crate::ir::node::{AxisRef, MultiIndex, Node, Site, SplitFactor, SplitList};

    fn lower(src: &str) -> Node {
        let expr = parse_expr(src).unwrap();
        lower_expr_to_node(&expr).unwrap()
    }

    #[test]
    fn lowers_pointwise_expr_with_canonical_axis_order() {
        let node = lower("ik*kj~ijk");

        assert_eq!(node.op, Op::Mul);
        assert_eq!(node.rank, 3);
        assert_eq!(
            node.inputs,
            vec![
                MultiIndex(vec![Index(0), Index(2)]),
                MultiIndex(vec![Index(2), Index(1)])
            ]
        );
        assert_eq!(node.output, MultiIndex(vec![Index(0), Index(1), Index(2)]));
        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
                AxisRef {
                    index: Index(2),
                    level: 0
                },
            ]
        );
        assert_eq!(node.compute_sites, vec![None, None]);
        assert_eq!(node.init_site, None);
    }

    #[test]
    fn lowers_repeated_input_axes() {
        let node = lower("+iij~i");

        assert_eq!(node.rank, 2);
        assert_eq!(
            node.inputs,
            vec![MultiIndex(vec![Index(0), Index(0), Index(1)])]
        );
        assert_eq!(node.output, MultiIndex(vec![Index(0)]));
        assert_eq!(
            node.init_site,
            Some(Site::At(AxisRef {
                index: Index(0),
                level: 0
            }))
        );
    }

    #[test]
    fn lowers_broadcasted_input_indexing() {
        let node = lower("ij+i~ij");

        assert_eq!(node.rank, 2);
        assert_eq!(
            node.inputs,
            vec![
                MultiIndex(vec![Index(0), Index(1)]),
                MultiIndex(vec![Index(0)])
            ]
        );
        assert_eq!(node.output, MultiIndex(vec![Index(0), Index(1)]));
        assert_eq!(node.init_site, None);
    }

    #[test]
    fn lowers_axes_in_output_first_order() {
        let node = lower("ji+i~ji");

        assert_eq!(node.rank, 2);
        assert_eq!(
            node.inputs,
            vec![
                MultiIndex(vec![Index(0), Index(1)]),
                MultiIndex(vec![Index(1)])
            ]
        );
        assert_eq!(node.output, MultiIndex(vec![Index(0), Index(1)]));
        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
            ]
        );
    }

    #[test]
    fn lowers_reduction_with_default_init_site() {
        let node = lower("+ijk~ij");

        assert_eq!(node.op, Op::Add);
        assert_eq!(node.rank, 3);
        assert_eq!(
            node.inputs,
            vec![MultiIndex(vec![Index(0), Index(1), Index(2)])]
        );
        assert_eq!(node.output, MultiIndex(vec![Index(0), Index(1)]));
        assert_eq!(
            node.init_site,
            Some(Site::At(AxisRef {
                index: Index(1),
                level: 0
            }))
        );
    }

    #[test]
    fn lowers_scalar_reduction_with_root_init_site() {
        let node = lower("+ij~");

        assert_eq!(node.rank, 2);
        assert_eq!(node.output, MultiIndex(vec![]));
        assert_eq!(node.init_site, Some(Site::Root));
    }

    #[test]
    fn lowers_split_order_and_compute_sites_from_permutation() {
        let node = lower("ik*kj~ijk|i:2,k:8|ik0i'k'1j");

        assert_eq!(
            node.splits,
            vec![
                SplitList(vec![SplitFactor(2)]),
                SplitList(vec![]),
                SplitList(vec![SplitFactor(8)]),
            ]
        );
        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(2),
                    level: 0
                },
                AxisRef {
                    index: Index(0),
                    level: 1
                },
                AxisRef {
                    index: Index(2),
                    level: 1
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
            ]
        );
        assert_eq!(
            node.compute_sites,
            vec![
                Some(Site::At(AxisRef {
                    index: Index(2),
                    level: 0,
                })),
                Some(Site::At(AxisRef {
                    index: Index(2),
                    level: 1,
                })),
            ]
        );
        assert_eq!(node.init_site, None);
    }

    #[test]
    fn lowers_reduction_with_explicit_bang() {
        let node = lower("+ijk~ij||ij!k");

        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
                AxisRef {
                    index: Index(2),
                    level: 0
                },
            ]
        );
        assert_eq!(
            node.init_site,
            Some(Site::At(AxisRef {
                index: Index(1),
                level: 0
            }))
        );
    }

    #[test]
    fn lowers_scalar_reduction_with_explicit_root_bang() {
        let node = lower("+ij~||!ij");

        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
            ]
        );
        assert_eq!(node.init_site, Some(Site::Root));
    }

    #[test]
    fn lowers_default_order_when_only_splits_are_present() {
        let node = lower("+ijk~ij|i:2,k:4|");

        assert_eq!(
            node.order,
            vec![
                AxisRef {
                    index: Index(0),
                    level: 0
                },
                AxisRef {
                    index: Index(0),
                    level: 1
                },
                AxisRef {
                    index: Index(1),
                    level: 0
                },
                AxisRef {
                    index: Index(2),
                    level: 0
                },
                AxisRef {
                    index: Index(2),
                    level: 1
                },
            ]
        );
        assert_eq!(
            node.init_site,
            Some(Site::At(AxisRef {
                index: Index(1),
                level: 0
            }))
        );
    }

    #[test]
    fn rejects_pointwise_bang() {
        let expr = parse_expr("ij~ij||ij!").unwrap();
        let error = lower_expr_to_node(&expr).unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: output init directive is only valid for reductions"
        );
    }

    #[test]
    fn rejects_interleaved_reduction_order_without_init_site_boundary() {
        let expr = parse_expr("+ijk~ij||ikj").unwrap();
        let error = lower_expr_to_node(&expr).unwrap_err();

        assert_eq!(
            error.to_string(),
            "reduction loops cannot appear before output loops complete"
        );
    }

    #[test]
    fn propagates_invalid_expr_errors() {
        let expr = parse_expr("ii~ii").unwrap();
        let error = lower_expr_to_node(&expr).unwrap_err();

        assert_eq!(error.to_string(), "expr 0: output pattern repeats axis `i`");
    }
}

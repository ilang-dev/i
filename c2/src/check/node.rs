use std::collections::BTreeSet;
use std::fmt;

use crate::ir::common::Index;
use crate::ir::node::{AxisRef, MultiIndex, Node, Site, SplitList};

pub fn validate_node(node: &Node) -> Result<(), ValidationError> {
    validate_node_shape(node)?;
    validate_schedule(node)
}

fn validate_node_shape(node: &Node) -> Result<(), ValidationError> {
    let mut referenced = BTreeSet::new();

    for (input_index, input) in node.inputs.iter().enumerate() {
        for index in &input.0 {
            validate_index(node.rank, *index)
                .map_err(|message| err(format!("input {} {}", input_index, message)))?;
            referenced.insert(index.0);
        }
    }

    let mut output_seen = BTreeSet::new();
    for index in &node.output.0 {
        validate_index(node.rank, *index).map_err(|message| err(format!("output {}", message)))?;
        if !output_seen.insert(index.0) {
            return Err(err(format!("output repeats index {}", index.0)));
        }
        referenced.insert(index.0);
    }

    for index in 0..node.rank {
        if !referenced.contains(&index) {
            return Err(err(format!("node index {} is not referenced", index)));
        }
    }

    Ok(())
}

fn validate_schedule(node: &Node) -> Result<(), ValidationError> {
    if node.splits.len() != node.rank {
        return Err(err(format!(
            "node has {} split lists for rank {}",
            node.splits.len(),
            node.rank
        )));
    }

    let expected = expected_axis_refs(node.splits.as_slice());
    let mut seen = BTreeSet::new();

    for axis_ref in &node.order {
        validate_axis_ref(node.rank, node.splits.as_slice(), *axis_ref)
            .map_err(|message| err(format!("order {}", message)))?;
        if !seen.insert((axis_ref.index.0, axis_ref.level)) {
            return Err(err(format!(
                "order repeats loop level {}",
                format_axis_ref(*axis_ref)
            )));
        }
    }

    if node.order.len() != expected.len() {
        return Err(err(format!(
            "order must contain {} loop levels, found {}",
            expected.len(),
            node.order.len()
        )));
    }

    for axis_ref in &expected {
        if !seen.contains(&(axis_ref.index.0, axis_ref.level)) {
            return Err(err(format!(
                "order is missing loop level {}",
                format_axis_ref(*axis_ref)
            )));
        }
    }

    if node.compute_sites.len() != node.inputs.len() {
        return Err(err(format!(
            "node has {} compute sites for {} inputs",
            node.compute_sites.len(),
            node.inputs.len()
        )));
    }

    for (input_index, site) in node.compute_sites.iter().enumerate() {
        if let Some(site) = site {
            validate_site(node.rank, node.splits.as_slice(), *site).map_err(|message| {
                err(format!(
                    "compute site for input {} {}",
                    input_index, message
                ))
            })?;
        }
    }

    let required_init_site = required_init_site_from_order(
        node.rank,
        &node.output,
        node.splits.as_slice(),
        &node.order,
    )?;
    match (required_init_site, node.init_site) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(err("pointwise node cannot have an init site")),
        (Some(_), None) => Err(err("reduction node must have an init site")),
        (Some(expected), Some(actual)) if expected == actual => Ok(()),
        (Some(expected), Some(_)) => {
            Err(err(format!("init site must be {}", format_site(expected))))
        }
    }
}

pub(crate) fn required_init_site_from_order(
    rank: usize,
    output: &MultiIndex,
    splits: &[SplitList],
    order: &[AxisRef],
) -> Result<Option<Site>, ValidationError> {
    let output_indexes = output
        .0
        .iter()
        .map(|index| index.0)
        .collect::<BTreeSet<_>>();
    let has_reduction = (0..rank).any(|index| !output_indexes.contains(&index));
    if !has_reduction {
        return Ok(None);
    }

    let output_levels = output
        .0
        .iter()
        .map(|index| splits[index.0].0.len() + 1)
        .sum::<usize>();
    if output_levels == 0 {
        return Ok(Some(Site::Root));
    }

    let mut seen_output_levels = 0usize;
    let mut saw_reduction = false;
    let mut last_output = None;

    for axis_ref in order {
        if output_indexes.contains(&axis_ref.index.0) {
            if saw_reduction {
                return Err(err(
                    "reduction loops cannot appear before output loops complete",
                ));
            }
            seen_output_levels += 1;
            last_output = Some(*axis_ref);
        } else {
            saw_reduction = true;
        }
    }

    if seen_output_levels != output_levels {
        return Err(err("order does not contain every output loop level"));
    }

    Ok(Some(Site::At(last_output.expect(
        "reduction with output levels must see one output loop level",
    ))))
}

fn expected_axis_refs(splits: &[SplitList]) -> Vec<AxisRef> {
    let mut expected = Vec::new();
    for (index, split_list) in splits.iter().enumerate() {
        for level in 0..=split_list.0.len() {
            expected.push(AxisRef {
                index: Index(index),
                level,
            });
        }
    }
    expected
}

fn validate_site(rank: usize, splits: &[SplitList], site: Site) -> Result<(), String> {
    match site {
        Site::Root => Ok(()),
        Site::At(axis_ref) => validate_axis_ref(rank, splits, axis_ref),
    }
}

fn validate_axis_ref(rank: usize, splits: &[SplitList], axis_ref: AxisRef) -> Result<(), String> {
    validate_index(rank, axis_ref.index)?;
    let max_level = splits[axis_ref.index.0].0.len();
    if axis_ref.level > max_level {
        return Err(format!(
            "references nonexistent loop level {}",
            format_axis_ref(axis_ref)
        ));
    }
    Ok(())
}

fn validate_index(rank: usize, index: Index) -> Result<(), String> {
    if index.0 >= rank {
        return Err(format!("references nonexistent index {}", index.0));
    }
    Ok(())
}

fn format_site(site: Site) -> String {
    match site {
        Site::Root => "Root".to_string(),
        Site::At(axis_ref) => format!("At({})", format_axis_ref(axis_ref)),
    }
}

fn format_axis_ref(axis_ref: AxisRef) -> String {
    format!("{}{}", axis_ref.index.0, "'".repeat(axis_ref.level))
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
    use super::validate_node;
    use crate::ir::common::Index;
    use crate::ir::common::Op;
    use crate::ir::node::{AxisRef, MultiIndex, Node, Site, SplitFactor, SplitList};

    #[test]
    fn accepts_pointwise_node() {
        let node = Node {
            op: Op::Add,
            rank: 2,
            inputs: vec![
                MultiIndex(vec![Index(0), Index(1)]),
                MultiIndex(vec![Index(0)]),
            ],
            output: MultiIndex(vec![Index(0), Index(1)]),
            splits: vec![SplitList(vec![]), SplitList(vec![])],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(1),
                    level: 0,
                },
            ],
            compute_sites: vec![
                None,
                Some(Site::At(AxisRef {
                    index: Index(0),
                    level: 0,
                })),
            ],
            init_site: None,
        };

        assert!(validate_node(&node).is_ok());
    }

    #[test]
    fn accepts_reduction_node() {
        let node = Node {
            op: Op::Add,
            rank: 3,
            inputs: vec![MultiIndex(vec![Index(0), Index(1), Index(2)])],
            output: MultiIndex(vec![Index(0), Index(1)]),
            splits: vec![
                SplitList(vec![SplitFactor(4)]),
                SplitList(vec![]),
                SplitList(vec![]),
            ],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(0),
                    level: 1,
                },
                AxisRef {
                    index: Index(1),
                    level: 0,
                },
                AxisRef {
                    index: Index(2),
                    level: 0,
                },
            ],
            compute_sites: vec![None],
            init_site: Some(Site::At(AxisRef {
                index: Index(1),
                level: 0,
            })),
        };

        assert!(validate_node(&node).is_ok());
    }

    #[test]
    fn rejects_split_len_mismatch() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 1,
            inputs: vec![MultiIndex(vec![Index(0)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![],
            order: vec![AxisRef {
                index: Index(0),
                level: 0,
            }],
            compute_sites: vec![None],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "node has 0 split lists for rank 1");
    }

    #[test]
    fn rejects_missing_loop_level() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 1,
            inputs: vec![MultiIndex(vec![Index(0)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![SplitFactor(4)])],
            order: vec![AxisRef {
                index: Index(0),
                level: 0,
            }],
            compute_sites: vec![None],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "order must contain 2 loop levels, found 1"
        );
    }

    #[test]
    fn rejects_duplicate_loop_level() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 1,
            inputs: vec![MultiIndex(vec![Index(0)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![])],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
            ],
            compute_sites: vec![None],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "order repeats loop level 0");
    }

    #[test]
    fn rejects_invalid_compute_site() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 1,
            inputs: vec![MultiIndex(vec![Index(0)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![])],
            order: vec![AxisRef {
                index: Index(0),
                level: 0,
            }],
            compute_sites: vec![Some(Site::At(AxisRef {
                index: Index(0),
                level: 1,
            }))],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "compute site for input 0 references nonexistent loop level 0'"
        );
    }

    #[test]
    fn rejects_out_of_range_index_in_multi_index() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 2,
            inputs: vec![MultiIndex(vec![Index(0), Index(2)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![]), SplitList(vec![])],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(1),
                    level: 0,
                },
            ],
            compute_sites: vec![None],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "input 0 references nonexistent index 2");
    }

    #[test]
    fn rejects_pointwise_init_site() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 1,
            inputs: vec![MultiIndex(vec![Index(0)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![])],
            order: vec![AxisRef {
                index: Index(0),
                level: 0,
            }],
            compute_sites: vec![None],
            init_site: Some(Site::Root),
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "pointwise node cannot have an init site");
    }

    #[test]
    fn rejects_reduction_without_init_site() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 2,
            inputs: vec![MultiIndex(vec![Index(0), Index(1)])],
            output: MultiIndex(vec![Index(0)]),
            splits: vec![SplitList(vec![]), SplitList(vec![])],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(1),
                    level: 0,
                },
            ],
            compute_sites: vec![None],
            init_site: None,
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "reduction node must have an init site");
    }

    #[test]
    fn rejects_interleaved_reduction_order() {
        let error = validate_node(&Node {
            op: Op::Add,
            rank: 3,
            inputs: vec![MultiIndex(vec![Index(0), Index(1), Index(2)])],
            output: MultiIndex(vec![Index(0), Index(2)]),
            splits: vec![SplitList(vec![]), SplitList(vec![]), SplitList(vec![])],
            order: vec![
                AxisRef {
                    index: Index(0),
                    level: 0,
                },
                AxisRef {
                    index: Index(1),
                    level: 0,
                },
                AxisRef {
                    index: Index(2),
                    level: 0,
                },
            ],
            compute_sites: vec![None],
            init_site: Some(Site::At(AxisRef {
                index: Index(2),
                level: 0,
            })),
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "reduction loops cannot appear before output loops complete"
        );
    }
}

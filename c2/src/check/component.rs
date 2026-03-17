use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::common::{Axis, AxisRef, Op, Pattern, Split};
use crate::ir::component::{Component, Expr};

pub fn validate_component(component: &Component) -> Result<(), ValidationError> {
    validate_component_inner(component)
}

fn validate_component_inner(component: &Component) -> Result<(), ValidationError> {
    match component {
        Component::Expr(expr) => validate_expr(expr),
        Component::Compose(left, right)
        | Component::Chain(left, right)
        | Component::Fanout(left, right)
        | Component::Pair(left, right) => {
            validate_component_inner(left)?;
            validate_component_inner(right)
        }
        Component::Swap(inner) => validate_component_inner(inner),
    }
}

fn validate_expr(expr: &Expr) -> Result<(), ValidationError> {
    for (input_index, input) in expr.inputs.iter().enumerate() {
        validate_pattern_chars(expr, PatternKind::Input(input_index), input)?;
    }
    validate_pattern_chars(expr, PatternKind::Output, &expr.output)?;
    validate_output_uniqueness(expr)?;

    let input_axes = collect_input_axes(expr);
    let output_axes = collect_axes(&expr.output);
    validate_form(expr, &input_axes, &output_axes)?;
    validate_schedule(expr, &input_axes, &output_axes)
}

fn validate_pattern_chars(
    expr: &Expr,
    pattern_kind: PatternKind,
    pattern: &Pattern,
) -> Result<(), ValidationError> {
    for axis in &pattern.0 {
        if !axis.is_ascii_lowercase() {
            return Err(err(
                expr,
                format!("{pattern_kind} uses invalid axis `{axis}`; axes must be [a-z]"),
            ));
        }
    }
    Ok(())
}

fn validate_output_uniqueness(expr: &Expr) -> Result<(), ValidationError> {
    let mut seen = BTreeSet::new();
    for axis in &expr.output.0 {
        if !seen.insert(*axis) {
            return Err(err(expr, format!("output pattern repeats axis `{axis}`")));
        }
    }
    Ok(())
}

fn validate_form(
    expr: &Expr,
    input_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    let is_reduction = input_axes != output_axes;

    match expr.inputs.len() {
        1 => validate_unary_form(expr, input_axes, output_axes, is_reduction),
        2 => validate_binary_form(expr, input_axes, output_axes, is_reduction),
        n => Err(err(
            expr,
            format!(
                "operator `{:?}` expects one or two inputs, found {n}",
                expr.op
            ),
        )),
    }
}

fn validate_unary_form(
    expr: &Expr,
    input_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
    is_reduction: bool,
) -> Result<(), ValidationError> {
    if !output_axes.is_subset(input_axes) {
        let axis = output_axes
            .difference(input_axes)
            .next()
            .copied()
            .expect("difference is non-empty");
        return Err(err(
            expr,
            format!("output axis `{axis}` does not appear in the input"),
        ));
    }

    match expr.op {
        Op::Not => {
            if is_reduction {
                return Err(err(
                    expr,
                    "operator `Not` is only valid in unary pointwise form",
                ));
            }
            Ok(())
        }
        op if is_reduction && !is_reducible(op) => Err(err(
            expr,
            format!("operator `{:?}` is not reducible", expr.op),
        )),
        _ => Ok(()),
    }
}

fn validate_binary_form(
    expr: &Expr,
    _input_axes: &BTreeSet<Axis>,
    _output_axes: &BTreeSet<Axis>,
    is_reduction: bool,
) -> Result<(), ValidationError> {
    if matches!(expr.op, Op::Not) {
        return Err(err(
            expr,
            "operator `Not` is only valid in unary pointwise form",
        ));
    }

    if is_reduction {
        return Err(err(
            expr,
            format!(
                "binary operator `{:?}` must be pointwise; output axes must match the union of input axes",
                expr.op
            ),
        ));
    }

    Ok(())
}

fn validate_schedule(
    expr: &Expr,
    input_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    let local_axes = input_axes
        .union(output_axes)
        .copied()
        .collect::<BTreeSet<_>>();
    let split_parts = validate_splits(expr, &local_axes)?;
    validate_order(expr, &local_axes, &split_parts)?;
    validate_compute_at(expr, &local_axes, &split_parts)?;
    Ok(())
}

fn validate_splits(
    expr: &Expr,
    local_axes: &BTreeSet<Axis>,
) -> Result<BTreeMap<Axis, usize>, ValidationError> {
    let mut split_defs = BTreeMap::new();
    let mut split_parts = BTreeMap::new();

    for split in &expr.schedule.splits {
        validate_split(expr, split, local_axes)?;
        match split_defs.get(&split.axis) {
            Some(existing_factors) if *existing_factors == split.factors => {
                return Err(err(
                    expr,
                    format!("duplicate split entry for axis `{}`", split.axis),
                ));
            }
            Some(_) => {
                return Err(err(
                    expr,
                    format!("contradictory split entries for axis `{}`", split.axis),
                ));
            }
            None => {
                split_defs.insert(split.axis, split.factors.clone());
                split_parts.insert(split.axis, split.factors.len());
            }
        }
    }

    Ok(split_parts)
}

fn validate_split(
    expr: &Expr,
    split: &Split,
    local_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    if !split.axis.is_ascii_lowercase() {
        return Err(err(
            expr,
            format!(
                "split uses invalid axis `{}`; axes must be [a-z]",
                split.axis
            ),
        ));
    }

    if !local_axes.contains(&split.axis) {
        return Err(err(
            expr,
            format!("split references unknown local axis `{}`", split.axis),
        ));
    }

    Ok(())
}

fn validate_order(
    expr: &Expr,
    local_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> Result<(), ValidationError> {
    if expr.schedule.order.is_empty() {
        return Ok(());
    }

    let expected = expected_axis_refs(local_axes, split_parts);
    let mut seen = BTreeSet::new();

    for axis_ref in &expr.schedule.order {
        validate_axis_ref(expr, "order", axis_ref, local_axes, split_parts)?;
        if !seen.insert((axis_ref.axis, axis_ref.part)) {
            return Err(err(
                expr,
                format!(
                    "order repeats axis part `{}{}`",
                    axis_ref.axis,
                    apostrophes(axis_ref.part)
                ),
            ));
        }
    }

    if seen.len() != expected.len() {
        return Err(err(
            expr,
            format!(
                "order must list each local axis part exactly once; expected {}, found {}",
                expected.len(),
                seen.len()
            ),
        ));
    }

    if seen != expected {
        let missing = expected
            .difference(&seen)
            .next()
            .copied()
            .expect("expected and seen differ");
        return Err(err(
            expr,
            format!(
                "order is missing axis part `{}{}`",
                missing.0,
                apostrophes(missing.1)
            ),
        ));
    }

    Ok(())
}

fn validate_compute_at(
    expr: &Expr,
    local_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> Result<(), ValidationError> {
    let compute_at = &expr.schedule.compute_at;
    if compute_at.is_empty() {
        return Ok(());
    }

    if compute_at.len() != expr.inputs.len() {
        return Err(err(
            expr,
            format!(
                "compute_at must have one entry per input when present; expected {}, found {}",
                expr.inputs.len(),
                compute_at.len()
            ),
        ));
    }

    for axis_ref in compute_at.iter().flatten() {
        validate_axis_ref(expr, "compute_at", axis_ref, local_axes, split_parts)?;
    }

    Ok(())
}

fn validate_axis_ref(
    expr: &Expr,
    context: &str,
    axis_ref: &AxisRef,
    local_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> Result<(), ValidationError> {
    if !axis_ref.axis.is_ascii_lowercase() {
        return Err(err(
            expr,
            format!(
                "{context} uses invalid axis `{}`; axes must be [a-z]",
                axis_ref.axis
            ),
        ));
    }

    if !local_axes.contains(&axis_ref.axis) {
        return Err(err(
            expr,
            format!(
                "{context} references unknown local axis `{}`",
                axis_ref.axis
            ),
        ));
    }

    let max_part = split_parts.get(&axis_ref.axis).copied().unwrap_or(0);
    if axis_ref.part > max_part {
        return Err(err(
            expr,
            format!(
                "{context} references nonexistent split part `{}{}`",
                axis_ref.axis,
                apostrophes(axis_ref.part)
            ),
        ));
    }

    Ok(())
}

fn expected_axis_refs(
    local_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> BTreeSet<(Axis, usize)> {
    let mut expected = BTreeSet::new();
    for axis in local_axes {
        let max_part = split_parts.get(axis).copied().unwrap_or(0);
        for part in 0..=max_part {
            expected.insert((*axis, part));
        }
    }
    expected
}

fn collect_input_axes(expr: &Expr) -> BTreeSet<Axis> {
    let mut axes = BTreeSet::new();
    for input in &expr.inputs {
        axes.extend(input.0.iter().copied());
    }
    axes
}

fn collect_axes(pattern: &Pattern) -> BTreeSet<Axis> {
    pattern.0.iter().copied().collect()
}

fn is_reducible(op: Op) -> bool {
    matches!(
        op,
        Op::Add | Op::Mul | Op::Max | Op::Min | Op::And | Op::Or | Op::Xor
    )
}

fn apostrophes(part: usize) -> String {
    "'".repeat(part)
}

fn err(expr: &Expr, message: impl Into<String>) -> ValidationError {
    ValidationError {
        expr: expr.id,
        message: message.into(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub expr: usize,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expr {}: {}", self.expr, self.message)
    }
}

impl std::error::Error for ValidationError {}

#[derive(Clone, Copy)]
enum PatternKind {
    Input(usize),
    Output,
}

impl fmt::Display for PatternKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(index) => write!(f, "input {}", index),
            Self::Output => write!(f, "output"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_component;
    use crate::ir::common::{AxisRef, Op, Pattern, Split};
    use crate::ir::component::{Component, Expr, Schedule};

    #[test]
    fn accepts_binary_pointwise_with_broadcast_and_schedule() {
        let expr = Expr {
            id: 0,
            op: Op::Add,
            inputs: vec![pattern("ij"), pattern("i")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![Split {
                    axis: 'j',
                    factors: vec![],
                }],
                order: vec![axis_ref('i', 0), axis_ref('j', 0)],
                compute_at: vec![],
            },
        };

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_unary_reduction_and_repeated_input_axes() {
        let expr = Expr {
            id: 0,
            op: Op::Add,
            inputs: vec![pattern("iij")],
            output: pattern("i"),
            schedule: Schedule {
                splits: vec![Split {
                    axis: 'j',
                    factors: vec![],
                }],
                order: vec![axis_ref('i', 0), axis_ref('j', 0)],
                compute_at: vec![None],
            },
        };

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn rejects_nonreducible_unary_reduction() {
        let error = validate_component(&Component::Expr(Expr {
            id: 7,
            op: Op::Sub,
            inputs: vec![pattern("ij")],
            output: pattern("i"),
            schedule: Schedule::default(),
        }))
        .unwrap_err();

        assert_eq!(error.to_string(), "expr 7: operator `Sub` is not reducible");
    }

    #[test]
    fn rejects_binary_reduction() {
        let error = validate_component(&Component::Expr(Expr {
            id: 3,
            op: Op::Add,
            inputs: vec![pattern("ij"), pattern("j")],
            output: pattern("i"),
            schedule: Schedule::default(),
        }))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 3: binary operator `Add` must be pointwise; output axes must match the union of input axes"
        );
    }

    #[test]
    fn rejects_not_in_binary_form() {
        let error = validate_component(&Component::Expr(Expr {
            id: 2,
            op: Op::Not,
            inputs: vec![pattern("i"), pattern("i")],
            output: pattern("i"),
            schedule: Schedule::default(),
        }))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 2: operator `Not` is only valid in unary pointwise form"
        );
    }

    #[test]
    fn rejects_output_axis_not_present_in_inputs() {
        let error = validate_component(&Component::Expr(Expr {
            id: 4,
            op: Op::Mul,
            inputs: vec![pattern("ij")],
            output: pattern("ik"),
            schedule: Schedule::default(),
        }))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 4: output axis `k` does not appear in the input"
        );
    }

    #[test]
    fn rejects_repeated_output_axes() {
        let error = validate_component(&Component::Expr(Expr {
            id: 1,
            op: Op::Add,
            inputs: vec![pattern("i")],
            output: pattern("ii"),
            schedule: Schedule::default(),
        }))
        .unwrap_err();

        assert_eq!(error.to_string(), "expr 1: output pattern repeats axis `i`");
    }

    #[test]
    fn rejects_duplicate_and_conflicting_splits() {
        let duplicate = validate_component(&Component::Expr(Expr {
            id: 5,
            op: Op::Add,
            inputs: vec![pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![
                    Split {
                        axis: 'i',
                        factors: vec![],
                    },
                    Split {
                        axis: 'i',
                        factors: vec![],
                    },
                ],
                order: vec![],
                compute_at: vec![],
            },
        }))
        .unwrap_err();

        assert_eq!(
            duplicate.to_string(),
            "expr 5: duplicate split entry for axis `i`"
        );

        let conflicting = validate_component(&Component::Expr(Expr {
            id: 6,
            op: Op::Add,
            inputs: vec![pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![
                    Split {
                        axis: 'i',
                        factors: vec![crate::ir::common::Extent::Known(4)],
                    },
                    Split {
                        axis: 'i',
                        factors: vec![
                            crate::ir::common::Extent::Known(4),
                            crate::ir::common::Extent::Known(8),
                        ],
                    },
                ],
                order: vec![],
                compute_at: vec![],
            },
        }))
        .unwrap_err();

        assert_eq!(
            conflicting.to_string(),
            "expr 6: contradictory split entries for axis `i`"
        );

        let same_arity_conflict = validate_component(&Component::Expr(Expr {
            id: 12,
            op: Op::Add,
            inputs: vec![pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![
                    Split {
                        axis: 'i',
                        factors: vec![crate::ir::common::Extent::Known(4)],
                    },
                    Split {
                        axis: 'i',
                        factors: vec![crate::ir::common::Extent::Known(8)],
                    },
                ],
                order: vec![],
                compute_at: vec![],
            },
        }))
        .unwrap_err();

        assert_eq!(
            same_arity_conflict.to_string(),
            "expr 12: contradictory split entries for axis `i`"
        );
    }

    #[test]
    fn rejects_nonexistent_order_part_and_incomplete_order() {
        let bad_part = validate_component(&Component::Expr(Expr {
            id: 8,
            op: Op::Add,
            inputs: vec![pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![Split {
                    axis: 'i',
                    factors: vec![crate::ir::common::Extent::Known(4)],
                }],
                order: vec![axis_ref('i', 2), axis_ref('j', 0)],
                compute_at: vec![],
            },
        }))
        .unwrap_err();

        assert_eq!(
            bad_part.to_string(),
            "expr 8: order references nonexistent split part `i''`"
        );

        let missing = validate_component(&Component::Expr(Expr {
            id: 9,
            op: Op::Add,
            inputs: vec![pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![Split {
                    axis: 'i',
                    factors: vec![crate::ir::common::Extent::Known(4)],
                }],
                order: vec![axis_ref('i', 0), axis_ref('j', 0)],
                compute_at: vec![],
            },
        }))
        .unwrap_err();

        assert_eq!(
            missing.to_string(),
            "expr 9: order must list each local axis part exactly once; expected 3, found 2"
        );
    }

    #[test]
    fn rejects_compute_at_length_mismatch_and_unknown_axis() {
        let mismatch = validate_component(&Component::Expr(Expr {
            id: 10,
            op: Op::Add,
            inputs: vec![pattern("ij"), pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![],
                order: vec![],
                compute_at: vec![None],
            },
        }))
        .unwrap_err();

        assert_eq!(
            mismatch.to_string(),
            "expr 10: compute_at must have one entry per input when present; expected 2, found 1"
        );

        let unknown_axis = validate_component(&Component::Expr(Expr {
            id: 11,
            op: Op::Add,
            inputs: vec![pattern("ij"), pattern("ij")],
            output: pattern("ij"),
            schedule: Schedule {
                splits: vec![],
                order: vec![],
                compute_at: vec![Some(axis_ref('k', 0)), None],
            },
        }))
        .unwrap_err();

        assert_eq!(
            unknown_axis.to_string(),
            "expr 11: compute_at references unknown local axis `k`"
        );
    }

    #[test]
    fn validates_nested_component_trees() {
        let component = Component::Chain(
            Box::new(Component::Expr(Expr {
                id: 0,
                op: Op::Mul,
                inputs: vec![pattern("ik"), pattern("kj")],
                output: pattern("ijk"),
                schedule: Schedule::default(),
            })),
            Box::new(Component::Swap(Box::new(Component::Pair(
                Box::new(Component::Expr(Expr {
                    id: 1,
                    op: Op::Add,
                    inputs: vec![pattern("ijk")],
                    output: pattern("ij"),
                    schedule: Schedule {
                        splits: vec![],
                        order: vec![axis_ref('i', 0), axis_ref('j', 0), axis_ref('k', 0)],
                        compute_at: vec![Some(axis_ref('j', 0))],
                    },
                })),
                Box::new(Component::Expr(Expr {
                    id: 2,
                    op: Op::Not,
                    inputs: vec![pattern("ij")],
                    output: pattern("ij"),
                    schedule: Schedule::default(),
                })),
            )))),
        );

        assert!(validate_component(&component).is_ok());
    }

    fn pattern(src: &str) -> Pattern {
        Pattern(src.chars().collect())
    }

    fn axis_ref(axis: char, part: usize) -> AxisRef {
        AxisRef { axis, part }
    }
}

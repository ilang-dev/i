use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::common::{Axis, AxisRef, Op};
use crate::ir::component::{Component, Expr, Operand, PermutationAtom};

pub fn validate_component(component: &Component) -> Result<(), ValidationError> {
    let mut next_expr = 0usize;
    validate_component_inner(component, &mut next_expr)
}

fn validate_component_inner(
    component: &Component,
    next_expr: &mut usize,
) -> Result<(), ValidationError> {
    match component {
        Component::Expr(expr) => {
            let expr_index = *next_expr;
            *next_expr += 1;
            validate_expr(expr, expr_index)
        }
        Component::Compose(left, right)
        | Component::Chain(left, right)
        | Component::Fanout(left, right)
        | Component::Pair(left, right) => {
            validate_component_inner(left, next_expr)?;
            validate_component_inner(right, next_expr)
        }
        Component::Swap(inner) => validate_component_inner(inner, next_expr),
    }
}

fn validate_expr(expr: &Expr, expr_index: usize) -> Result<(), ValidationError> {
    for (input_index, input) in expr.inputs.iter().enumerate() {
        validate_pattern_chars(expr_index, PatternKind::Input(input_index), input)?;
    }
    validate_pattern_chars(expr_index, PatternKind::Output, &expr.output)?;
    validate_output_uniqueness(expr, expr_index)?;

    let input_axes = collect_input_axes(expr);
    let output_axes = collect_axes(&expr.output);
    validate_form(expr, expr_index, &input_axes, &output_axes)?;
    validate_schedule(expr, expr_index, &input_axes, &output_axes)
}

fn validate_pattern_chars(
    expr_index: usize,
    pattern_kind: PatternKind,
    pattern: &[char],
) -> Result<(), ValidationError> {
    for axis in pattern {
        if !axis.is_ascii_lowercase() {
            return Err(err(
                expr_index,
                format!("{pattern_kind} uses invalid axis `{axis}`; axes must be [a-z]"),
            ));
        }
    }
    Ok(())
}

fn validate_output_uniqueness(expr: &Expr, expr_index: usize) -> Result<(), ValidationError> {
    let mut seen = BTreeSet::new();
    for axis in &expr.output {
        if !seen.insert(*axis) {
            return Err(err(
                expr_index,
                format!("output pattern repeats axis `{axis}`"),
            ));
        }
    }
    Ok(())
}

fn validate_form(
    expr: &Expr,
    expr_index: usize,
    input_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    let is_reduction = input_axes != output_axes;

    match expr.inputs.len() {
        1 => validate_unary_form(expr, expr_index, input_axes, output_axes, is_reduction),
        2 => validate_binary_form(expr, expr_index, input_axes, output_axes, is_reduction),
        n => Err(err(
            expr_index,
            format!(
                "operator `{:?}` expects one or two inputs, found {n}",
                expr.op
            ),
        )),
    }
}

fn validate_unary_form(
    expr: &Expr,
    expr_index: usize,
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
            expr_index,
            format!("output axis `{axis}` does not appear in the input"),
        ));
    }

    match expr.op {
        Op::Not => {
            if is_reduction {
                return Err(err(
                    expr_index,
                    "operator `Not` is only valid in unary pointwise form",
                ));
            }
            Ok(())
        }
        op if is_reduction && !is_reducible(op) => Err(err(
            expr_index,
            format!("operator `{:?}` is not reducible", expr.op),
        )),
        _ => Ok(()),
    }
}

fn validate_binary_form(
    expr: &Expr,
    expr_index: usize,
    _input_axes: &BTreeSet<Axis>,
    _output_axes: &BTreeSet<Axis>,
    is_reduction: bool,
) -> Result<(), ValidationError> {
    if matches!(expr.op, Op::Not) {
        return Err(err(
            expr_index,
            "operator `Not` is only valid in unary pointwise form",
        ));
    }

    if is_reduction {
        return Err(err(
            expr_index,
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
    expr_index: usize,
    input_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    let local_axes = input_axes
        .union(output_axes)
        .copied()
        .collect::<BTreeSet<_>>();
    let split_parts = validate_splits(expr, expr_index, &local_axes)?;
    validate_permutation(expr, expr_index, &local_axes, output_axes, &split_parts)?;
    Ok(())
}

fn validate_splits(
    expr: &Expr,
    expr_index: usize,
    local_axes: &BTreeSet<Axis>,
) -> Result<BTreeMap<Axis, usize>, ValidationError> {
    let mut split_parts = BTreeMap::new();

    for (axis, factors) in &expr.splits {
        validate_split(expr_index, *axis, local_axes)?;
        match split_parts.insert(*axis, factors.len()) {
            Some(existing) if existing == factors.len() => {
                return Err(err(
                    expr_index,
                    format!("duplicate split entry for axis `{axis}`"),
                ));
            }
            Some(_) => {
                return Err(err(
                    expr_index,
                    format!("contradictory split entries for axis `{axis}`"),
                ));
            }
            None => {}
        }
    }

    Ok(split_parts)
}

fn validate_split(
    expr_index: usize,
    axis: Axis,
    local_axes: &BTreeSet<Axis>,
) -> Result<(), ValidationError> {
    if !axis.is_ascii_lowercase() {
        return Err(err(
            expr_index,
            format!("split uses invalid axis `{axis}`; axes must be [a-z]"),
        ));
    }

    if !local_axes.contains(&axis) {
        return Err(err(
            expr_index,
            format!("split references unknown local axis `{axis}`"),
        ));
    }

    Ok(())
}

fn validate_permutation(
    expr: &Expr,
    expr_index: usize,
    local_axes: &BTreeSet<Axis>,
    output_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> Result<(), ValidationError> {
    if expr.permutation.is_empty() {
        return Ok(());
    }

    let expected = expected_axis_refs(local_axes, split_parts);
    let output_parts = expected_axis_refs(output_axes, split_parts);
    let mut seen = BTreeSet::new();
    let mut last_axis_ref = None;
    let mut seen_inputs = BTreeSet::new();
    let mut seen_bang = false;

    for atom in &expr.permutation {
        match atom {
            PermutationAtom::Axis { axis, part } => {
                let axis_ref = AxisRef {
                    axis: *axis,
                    part: *part,
                };
                validate_axis_ref(
                    expr_index,
                    "permutation",
                    &axis_ref,
                    local_axes,
                    split_parts,
                )?;
                if !seen.insert((axis_ref.axis, axis_ref.part)) {
                    return Err(err(
                        expr_index,
                        format!(
                            "permutation repeats axis part `{}{}`",
                            axis_ref.axis,
                            apostrophes(axis_ref.part)
                        ),
                    ));
                }
                last_axis_ref = Some(axis_ref);
            }
            PermutationAtom::Input(input) => {
                let input_index = operand_index(*input);
                if input_index >= expr.inputs.len() {
                    return Err(err(
                        expr_index,
                        format!("permutation references nonexistent input {}", input_index),
                    ));
                }
                if last_axis_ref.is_none() {
                    return Err(err(
                        expr_index,
                        format!(
                            "input {} cannot appear before any loop in permutation",
                            input_index
                        ),
                    ));
                }
                if !seen_inputs.insert(*input) {
                    return Err(err(
                        expr_index,
                        format!("duplicate compute directive for input {}", input_index),
                    ));
                }
            }
            PermutationAtom::Bang => {
                if seen_bang {
                    return Err(err(expr_index, "duplicate output init directive"));
                }
                if local_axes == output_axes {
                    return Err(err(
                        expr_index,
                        "output init directive is only valid for reductions",
                    ));
                }
                seen_bang = true;

                if let Some(missing) = output_parts.difference(&seen).next() {
                    return Err(err(
                        expr_index,
                        format!(
                            "output init cannot appear before output axis part `{}{}`",
                            missing.0,
                            apostrophes(missing.1)
                        ),
                    ));
                }
            }
        }
    }

    if seen.len() != expected.len() {
        return Err(err(
            expr_index,
            format!(
                "permutation must list each local axis part exactly once; expected {}, found {}",
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
            expr_index,
            format!(
                "permutation is missing axis part `{}{}`",
                missing.0,
                apostrophes(missing.1)
            ),
        ));
    }

    Ok(())
}
fn validate_axis_ref(
    expr_index: usize,
    context: &str,
    axis_ref: &AxisRef,
    local_axes: &BTreeSet<Axis>,
    split_parts: &BTreeMap<Axis, usize>,
) -> Result<(), ValidationError> {
    if !axis_ref.axis.is_ascii_lowercase() {
        return Err(err(
            expr_index,
            format!(
                "{context} uses invalid axis `{}`; axes must be [a-z]",
                axis_ref.axis
            ),
        ));
    }

    if !local_axes.contains(&axis_ref.axis) {
        return Err(err(
            expr_index,
            format!(
                "{context} references unknown local axis `{}`",
                axis_ref.axis
            ),
        ));
    }

    let max_part = split_parts.get(&axis_ref.axis).copied().unwrap_or(0);
    if axis_ref.part > max_part {
        return Err(err(
            expr_index,
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
        axes.extend(input.iter().copied());
    }
    axes
}

fn collect_axes(pattern: &[char]) -> BTreeSet<Axis> {
    pattern.iter().copied().collect()
}

fn is_reducible(op: Op) -> bool {
    matches!(
        op,
        Op::Add | Op::Mul | Op::Max | Op::Min | Op::And | Op::Or | Op::Xor
    )
}

fn operand_index(operand: Operand) -> usize {
    match operand {
        Operand::Left => 0,
        Operand::Right => 1,
    }
}

fn apostrophes(part: usize) -> String {
    "'".repeat(part)
}

fn err(expr: usize, message: impl Into<String>) -> ValidationError {
    ValidationError {
        expr,
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
    use crate::ir::common::Op;
    use crate::ir::component::{Component, Operand, PermutationAtom};
    use crate::ir::expr::Expr;
    #[test]
    fn accepts_binary_pointwise_with_broadcast_and_permutation() {
        let expr = make_expr(
            Op::Add,
            &["ij", "i"],
            "ij",
            vec![],
            vec![axis('i', 0), input(0), axis('j', 0), input(1)],
        );

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_unary_reduction_and_repeated_input_axes() {
        let expr = make_expr(
            Op::Add,
            &["iij"],
            "i",
            vec![],
            vec![axis('i', 0), axis('j', 0), input(0)],
        );

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_unary_not_pointwise() {
        let expr = make_expr(Op::Not, &["ij"], "ij", vec![], Vec::new());

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_scalar_reduction_with_root_bang() {
        let expr = make_expr(
            Op::Add,
            &["ij"],
            "",
            vec![],
            vec![bang(), axis('i', 0), axis('j', 0), input(0)],
        );

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_split_reduction_with_bang_after_output_parts() {
        let expr = make_expr(
            Op::Add,
            &["ik"],
            "i",
            vec![('i', vec![4])],
            vec![axis('i', 0), axis('i', 1), bang(), axis('k', 0), input(0)],
        );

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn accepts_split_reduction_with_split_reduction_axis() {
        let expr = make_expr(
            Op::Add,
            &["ijk"],
            "ij",
            vec![('k', vec![2])],
            vec![
                axis('i', 0),
                axis('j', 0),
                bang(),
                axis('k', 0),
                axis('k', 1),
            ],
        );

        assert!(validate_component(&Component::Expr(expr)).is_ok());
    }

    #[test]
    fn rejects_nonreducible_unary_reduction() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Sub,
            &["ij"],
            "i",
            vec![],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(error.to_string(), "expr 0: operator `Sub` is not reducible");
    }

    #[test]
    fn rejects_binary_reduction() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij", "j"],
            "i",
            vec![],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: binary operator `Add` must be pointwise; output axes must match the union of input axes"
        );
    }

    #[test]
    fn rejects_not_in_binary_form() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Not,
            &["i", "i"],
            "i",
            vec![],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: operator `Not` is only valid in unary pointwise form"
        );
    }

    #[test]
    fn rejects_output_axis_not_present_in_inputs() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Mul,
            &["ij"],
            "ik",
            vec![],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: output axis `k` does not appear in the input"
        );
    }

    #[test]
    fn rejects_repeated_output_axes() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["i"],
            "ii",
            vec![],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(error.to_string(), "expr 0: output pattern repeats axis `i`");
    }

    #[test]
    fn rejects_unknown_split_axis() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('k', vec![4])],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: split references unknown local axis `k`"
        );
    }

    #[test]
    fn rejects_duplicate_and_contradictory_split_entries() {
        let duplicate = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('i', vec![4]), ('i', vec![8])],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            duplicate.to_string(),
            "expr 0: duplicate split entry for axis `i`"
        );

        let contradictory = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('i', vec![4]), ('i', vec![2, 2])],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            contradictory.to_string(),
            "expr 0: contradictory split entries for axis `i`"
        );
    }

    #[test]
    fn rejects_bad_permutation_shape() {
        let repeated_axis = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![axis('i', 0), axis('j', 0), axis('i', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            repeated_axis.to_string(),
            "expr 0: permutation repeats axis part `i`"
        );

        let missing_axis = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('i', vec![4])],
            vec![axis('i', 0), axis('j', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            missing_axis.to_string(),
            "expr 0: permutation must list each local axis part exactly once; expected 3, found 2"
        );

        let bad_part = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('i', vec![4])],
            vec![axis('i', 0), axis('i', 2), axis('j', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            bad_part.to_string(),
            "expr 0: permutation references nonexistent split part `i''`"
        );
    }

    #[test]
    fn rejects_bad_permutation_axis_references() {
        let unknown_axis = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![axis('i', 0), axis('k', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            unknown_axis.to_string(),
            "expr 0: permutation references unknown local axis `k`"
        );

        let invalid_axis = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![axis('I', 0), axis('j', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            invalid_axis.to_string(),
            "expr 0: permutation uses invalid axis `I`; axes must be [a-z]"
        );
    }

    #[test]
    fn rejects_bad_input_directives() {
        let duplicate = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij", "ij"],
            "ij",
            vec![],
            vec![axis('i', 0), input(0), axis('j', 0), input(0)],
        )))
        .unwrap_err();

        assert_eq!(
            duplicate.to_string(),
            "expr 0: duplicate compute directive for input 0"
        );

        let no_loop = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![input(0), axis('i', 0), axis('j', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            no_loop.to_string(),
            "expr 0: input 0 cannot appear before any loop in permutation"
        );

        let nonexistent = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![axis('i', 0), input(1), axis('j', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            nonexistent.to_string(),
            "expr 0: permutation references nonexistent input 1"
        );
    }

    #[test]
    fn rejects_invalid_split_axis_char() {
        let error = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![('I', vec![4])],
            Vec::new(),
        )))
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "expr 0: split uses invalid axis `I`; axes must be [a-z]"
        );
    }

    #[test]
    fn rejects_bad_output_init_directives() {
        let duplicate = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ijk"],
            "ij",
            vec![],
            vec![axis('i', 0), axis('j', 0), bang(), bang(), axis('k', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            duplicate.to_string(),
            "expr 0: duplicate output init directive"
        );

        let before_output = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ijk"],
            "ij",
            vec![],
            vec![axis('i', 0), bang(), axis('j', 0), axis('k', 0)],
        )))
        .unwrap_err();

        assert_eq!(
            before_output.to_string(),
            "expr 0: output init cannot appear before output axis part `j`"
        );

        let inside_reduction = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ijk"],
            "ij",
            vec![],
            vec![axis('i', 0), axis('j', 0), axis('k', 0), bang()],
        )));

        assert!(inside_reduction.is_ok());

        let pointwise = validate_component(&Component::Expr(make_expr(
            Op::Add,
            &["ij"],
            "ij",
            vec![],
            vec![axis('i', 0), axis('j', 0), bang()],
        )))
        .unwrap_err();

        assert_eq!(
            pointwise.to_string(),
            "expr 0: output init directive is only valid for reductions"
        );
    }

    #[test]
    fn validates_nested_component_trees() {
        let component = Component::Chain(
            Box::new(Component::Expr(make_expr(
                Op::Mul,
                &["ik", "kj"],
                "ijk",
                vec![],
                Vec::new(),
            ))),
            Box::new(Component::Swap(Box::new(Component::Pair(
                Box::new(Component::Expr(make_expr(
                    Op::Add,
                    &["ijk"],
                    "ij",
                    vec![],
                    vec![axis('i', 0), axis('j', 0), input(0), axis('k', 0)],
                ))),
                Box::new(Component::Expr(make_expr(
                    Op::Not,
                    &["ij"],
                    "ij",
                    vec![],
                    Vec::new(),
                ))),
            )))),
        );

        assert!(validate_component(&component).is_ok());
    }

    fn make_expr(
        op: Op,
        inputs: &[&str],
        output: &str,
        splits: Vec<(char, Vec<usize>)>,
        permutation: Vec<PermutationAtom>,
    ) -> Expr {
        Expr {
            op,
            inputs: inputs.iter().map(|src| pattern(src)).collect(),
            output: pattern(output),
            splits,
            permutation,
        }
    }

    fn pattern(src: &str) -> Vec<char> {
        src.chars().collect()
    }

    fn axis(axis: char, part: usize) -> PermutationAtom {
        PermutationAtom::Axis { axis, part }
    }

    fn input(index: usize) -> PermutationAtom {
        match index {
            0 => PermutationAtom::Input(Operand::Left),
            1 => PermutationAtom::Input(Operand::Right),
            _ => panic!("invalid operand index"),
        }
    }

    fn bang() -> PermutationAtom {
        PermutationAtom::Bang
    }
}

use std::collections::BTreeMap;
use std::fmt;

use crate::check::component::validate_component;
use crate::check::stage::{required_init_site_from_order, validate_scheduled_stage};
use crate::ir::component::Component;
use crate::ir::expr::{Expr, PermutationAtom};
use crate::ir::stage::{
    Axis, AxisRef, Index, Schedule, ScheduledStage, Site, SplitFactor, SplitList, Stage,
};

pub fn lower_expr_to_stage(expr: &Expr) -> Result<ScheduledStage, LowerError> {
    validate_component(&Component::Expr(expr.clone())).map_err(LowerError::from_component)?;

    let axis_order = canonical_axis_order(expr);
    let axis_map = axis_order
        .iter()
        .enumerate()
        .map(|(index, axis)| (*axis, Axis(index)))
        .collect::<BTreeMap<_, _>>();

    let stage = Stage {
        op: expr.op,
        rank: axis_order.len(),
        inputs: expr
            .inputs
            .iter()
            .map(|pattern| lower_index(pattern, &axis_map))
            .collect::<Result<_, _>>()?,
        output: lower_index(&expr.output, &axis_map)?,
    };

    let splits = lower_splits(expr, &axis_order);
    let (order, compute_sites, init_site) = if expr.permutation.is_empty() {
        let order = default_order(splits.as_slice());
        let init_site = required_init_site_from_order(&stage, splits.as_slice(), &order)
            .map_err(LowerError::from_stage)?;
        (order, vec![None; stage.inputs.len()], init_site)
    } else {
        lower_schedule_from_permutation(expr, &stage, splits.as_slice(), &axis_map)?
    };

    let scheduled = ScheduledStage {
        stage,
        schedule: Schedule {
            splits,
            order,
            compute_sites,
            init_site,
        },
    };
    validate_scheduled_stage(&scheduled).map_err(LowerError::from_stage)?;
    Ok(scheduled)
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

fn lower_index(pattern: &[char], axis_map: &BTreeMap<char, Axis>) -> Result<Index, LowerError> {
    pattern
        .iter()
        .map(|axis| {
            axis_map
                .get(axis)
                .copied()
                .ok_or_else(|| LowerError::new(format!("unknown local axis `{axis}`")))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Index)
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
    for (axis, split_list) in splits.iter().enumerate() {
        for part in 0..=split_list.0.len() {
            order.push(AxisRef {
                axis: Axis(axis),
                part,
            });
        }
    }
    order
}

fn lower_schedule_from_permutation(
    expr: &Expr,
    stage: &Stage,
    splits: &[SplitList],
    axis_map: &BTreeMap<char, Axis>,
) -> Result<(Vec<AxisRef>, Vec<Option<Site>>, Option<Site>), LowerError> {
    let mut order = Vec::new();
    let mut compute_sites = vec![None; expr.inputs.len()];
    let mut last_axis_ref = None;
    let mut explicit_init_site = None;

    for atom in &expr.permutation {
        match atom {
            PermutationAtom::Axis { axis, part } => {
                let axis = axis_map
                    .get(axis)
                    .copied()
                    .ok_or_else(|| LowerError::new(format!("unknown local axis `{axis}`")))?;
                let axis_ref = AxisRef { axis, part: *part };
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
        required_init_site_from_order(stage, splits, &order).map_err(LowerError::from_stage)?;
    let init_site = match (required_init_site, explicit_init_site) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(LowerError::new(
                "pointwise stage cannot have an output init directive",
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
        Site::At(axis_ref) => format!("At({}{})", axis_ref.axis.0, "'".repeat(axis_ref.part)),
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

    fn from_stage(error: crate::check::stage::ValidationError) -> Self {
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
    use super::lower_expr_to_stage;
    use crate::front::parse_expr;
    use crate::ir::common::Op;
    use crate::ir::stage::{Axis, AxisRef, Index, Site, SplitFactor, SplitList};

    fn lower(src: &str) -> crate::ir::stage::ScheduledStage {
        let expr = parse_expr(src).unwrap();
        lower_expr_to_stage(&expr).unwrap()
    }

    #[test]
    fn lowers_pointwise_expr_with_canonical_axis_order() {
        let stage = lower("ik*kj~ijk");

        assert_eq!(stage.stage.op, Op::Mul);
        assert_eq!(stage.stage.rank, 3);
        assert_eq!(
            stage.stage.inputs,
            vec![Index(vec![Axis(0), Axis(2)]), Index(vec![Axis(2), Axis(1)])]
        );
        assert_eq!(stage.stage.output, Index(vec![Axis(0), Axis(1), Axis(2)]));
        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
                AxisRef {
                    axis: Axis(2),
                    part: 0
                },
            ]
        );
        assert_eq!(stage.schedule.compute_sites, vec![None, None]);
        assert_eq!(stage.schedule.init_site, None);
    }

    #[test]
    fn lowers_repeated_input_axes() {
        let stage = lower("+iij~i");

        assert_eq!(stage.stage.rank, 2);
        assert_eq!(
            stage.stage.inputs,
            vec![Index(vec![Axis(0), Axis(0), Axis(1)])]
        );
        assert_eq!(stage.stage.output, Index(vec![Axis(0)]));
        assert_eq!(
            stage.schedule.init_site,
            Some(Site::At(AxisRef {
                axis: Axis(0),
                part: 0
            }))
        );
    }

    #[test]
    fn lowers_broadcasted_input_indexing() {
        let stage = lower("ij+i~ij");

        assert_eq!(stage.stage.rank, 2);
        assert_eq!(
            stage.stage.inputs,
            vec![Index(vec![Axis(0), Axis(1)]), Index(vec![Axis(0)])]
        );
        assert_eq!(stage.stage.output, Index(vec![Axis(0), Axis(1)]));
        assert_eq!(stage.schedule.init_site, None);
    }

    #[test]
    fn lowers_axes_in_output_first_order() {
        let stage = lower("ji+i~ji");

        assert_eq!(stage.stage.rank, 2);
        assert_eq!(
            stage.stage.inputs,
            vec![Index(vec![Axis(0), Axis(1)]), Index(vec![Axis(1)])]
        );
        assert_eq!(stage.stage.output, Index(vec![Axis(0), Axis(1)]));
        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
            ]
        );
    }

    #[test]
    fn lowers_reduction_with_default_init_site() {
        let stage = lower("+ijk~ij");

        assert_eq!(stage.stage.op, Op::Add);
        assert_eq!(stage.stage.rank, 3);
        assert_eq!(
            stage.stage.inputs,
            vec![Index(vec![Axis(0), Axis(1), Axis(2)])]
        );
        assert_eq!(stage.stage.output, Index(vec![Axis(0), Axis(1)]));
        assert_eq!(
            stage.schedule.init_site,
            Some(Site::At(AxisRef {
                axis: Axis(1),
                part: 0
            }))
        );
    }

    #[test]
    fn lowers_scalar_reduction_with_root_init_site() {
        let stage = lower("+ij~");

        assert_eq!(stage.stage.rank, 2);
        assert_eq!(stage.stage.output, Index(vec![]));
        assert_eq!(stage.schedule.init_site, Some(Site::Root));
    }

    #[test]
    fn lowers_split_order_and_compute_sites_from_permutation() {
        let stage = lower("ik*kj~ijk|i:2,k:8|ik0i'k'1j");

        assert_eq!(
            stage.schedule.splits,
            vec![
                SplitList(vec![SplitFactor(2)]),
                SplitList(vec![]),
                SplitList(vec![SplitFactor(8)]),
            ]
        );
        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(2),
                    part: 0
                },
                AxisRef {
                    axis: Axis(0),
                    part: 1
                },
                AxisRef {
                    axis: Axis(2),
                    part: 1
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
            ]
        );
        assert_eq!(
            stage.schedule.compute_sites,
            vec![
                Some(Site::At(AxisRef {
                    axis: Axis(2),
                    part: 0,
                })),
                Some(Site::At(AxisRef {
                    axis: Axis(2),
                    part: 1,
                })),
            ]
        );
        assert_eq!(stage.schedule.init_site, None);
    }

    #[test]
    fn lowers_reduction_with_explicit_bang() {
        let stage = lower("+ijk~ij||ij!k");

        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
                AxisRef {
                    axis: Axis(2),
                    part: 0
                },
            ]
        );
        assert_eq!(
            stage.schedule.init_site,
            Some(Site::At(AxisRef {
                axis: Axis(1),
                part: 0
            }))
        );
    }

    #[test]
    fn lowers_scalar_reduction_with_explicit_root_bang() {
        let stage = lower("+ij~||!ij");

        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
            ]
        );
        assert_eq!(stage.schedule.init_site, Some(Site::Root));
    }

    #[test]
    fn lowers_default_order_when_only_splits_are_present() {
        let stage = lower("+ijk~ij|i:2,k:4|");

        assert_eq!(
            stage.schedule.order,
            vec![
                AxisRef {
                    axis: Axis(0),
                    part: 0
                },
                AxisRef {
                    axis: Axis(0),
                    part: 1
                },
                AxisRef {
                    axis: Axis(1),
                    part: 0
                },
                AxisRef {
                    axis: Axis(2),
                    part: 0
                },
                AxisRef {
                    axis: Axis(2),
                    part: 1
                },
            ]
        );
        assert_eq!(
            stage.schedule.init_site,
            Some(Site::At(AxisRef {
                axis: Axis(1),
                part: 0
            }))
        );
    }

    #[test]
    fn rejects_pointwise_bang() {
        let expr = parse_expr("ij~ij||ij!").unwrap();
        let error = lower_expr_to_stage(&expr).unwrap_err();

        assert_eq!(
            error.to_string(),
            "pointwise stage cannot have an output init directive"
        );
    }

    #[test]
    fn rejects_interleaved_reduction_order_without_init_site_boundary() {
        let expr = parse_expr("+ijk~ij||ikj").unwrap();
        let error = lower_expr_to_stage(&expr).unwrap_err();

        assert_eq!(
            error.to_string(),
            "reduction loops cannot appear before output loops complete"
        );
    }

    #[test]
    fn propagates_invalid_expr_errors() {
        let expr = parse_expr("ii~ii").unwrap();
        let error = lower_expr_to_stage(&expr).unwrap_err();

        assert_eq!(error.to_string(), "expr 0: output pattern repeats axis `i`");
    }
}

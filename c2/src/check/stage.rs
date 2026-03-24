use std::collections::BTreeSet;
use std::fmt;

use crate::ir::stage::{Axis, AxisRef, Schedule, ScheduledStage, Site, SplitList, Stage};

pub fn validate_scheduled_stage(stage: &ScheduledStage) -> Result<(), ValidationError> {
    validate_stage(&stage.stage)?;
    validate_schedule(&stage.stage, &stage.schedule)
}

fn validate_stage(stage: &Stage) -> Result<(), ValidationError> {
    let mut referenced = BTreeSet::new();

    for (input_index, input) in stage.inputs.iter().enumerate() {
        for axis in &input.0 {
            validate_axis(stage.rank, *axis)
                .map_err(|message| err(format!("input {} {}", input_index, message)))?;
            referenced.insert(axis.0);
        }
    }

    let mut output_seen = BTreeSet::new();
    for axis in &stage.output.0 {
        validate_axis(stage.rank, *axis).map_err(|message| err(format!("output {}", message)))?;
        if !output_seen.insert(axis.0) {
            return Err(err(format!("output repeats axis {}", axis.0)));
        }
        referenced.insert(axis.0);
    }

    for axis in 0..stage.rank {
        if !referenced.contains(&axis) {
            return Err(err(format!("stage axis {} is not referenced", axis)));
        }
    }

    Ok(())
}

fn validate_schedule(stage: &Stage, schedule: &Schedule) -> Result<(), ValidationError> {
    if schedule.splits.len() != stage.rank {
        return Err(err(format!(
            "schedule has {} split lists for rank {}",
            schedule.splits.len(),
            stage.rank
        )));
    }

    let expected = expected_axis_refs(schedule.splits.as_slice());
    let mut seen = BTreeSet::new();

    for axis_ref in &schedule.order {
        validate_axis_ref(stage.rank, schedule.splits.as_slice(), *axis_ref)
            .map_err(|message| err(format!("order {}", message)))?;
        if !seen.insert((axis_ref.axis.0, axis_ref.part)) {
            return Err(err(format!(
                "order repeats loop part {}",
                format_axis_ref(*axis_ref)
            )));
        }
    }

    if schedule.order.len() != expected.len() {
        return Err(err(format!(
            "order must contain {} loop parts, found {}",
            expected.len(),
            schedule.order.len()
        )));
    }

    for axis_ref in &expected {
        if !seen.contains(&(axis_ref.axis.0, axis_ref.part)) {
            return Err(err(format!(
                "order is missing loop part {}",
                format_axis_ref(*axis_ref)
            )));
        }
    }

    if schedule.compute_sites.len() != stage.inputs.len() {
        return Err(err(format!(
            "schedule has {} compute sites for {} inputs",
            schedule.compute_sites.len(),
            stage.inputs.len()
        )));
    }

    for (input_index, site) in schedule.compute_sites.iter().enumerate() {
        if let Some(site) = site {
            validate_site(stage.rank, schedule.splits.as_slice(), *site).map_err(|message| {
                err(format!(
                    "compute site for input {} {}",
                    input_index, message
                ))
            })?;
        }
    }

    let required_init_site =
        required_init_site_from_order(stage, schedule.splits.as_slice(), &schedule.order)?;
    match (required_init_site, schedule.init_site) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(err("pointwise stage cannot have an init site")),
        (Some(_), None) => Err(err("reduction stage must have an init site")),
        (Some(expected), Some(actual)) if expected == actual => Ok(()),
        (Some(expected), Some(_)) => {
            Err(err(format!("init site must be {}", format_site(expected))))
        }
    }
}

pub(crate) fn required_init_site_from_order(
    stage: &Stage,
    splits: &[SplitList],
    order: &[AxisRef],
) -> Result<Option<Site>, ValidationError> {
    let output_axes = stage
        .output
        .0
        .iter()
        .map(|axis| axis.0)
        .collect::<BTreeSet<_>>();
    let has_reduction = (0..stage.rank).any(|axis| !output_axes.contains(&axis));
    if !has_reduction {
        return Ok(None);
    }

    let output_parts = stage
        .output
        .0
        .iter()
        .map(|axis| splits[axis.0].0.len() + 1)
        .sum::<usize>();
    if output_parts == 0 {
        return Ok(Some(Site::Root));
    }

    let mut seen_output_parts = 0usize;
    let mut saw_reduction = false;
    let mut last_output = None;

    for axis_ref in order {
        if output_axes.contains(&axis_ref.axis.0) {
            if saw_reduction {
                return Err(err(
                    "reduction loops cannot appear before output loops complete",
                ));
            }
            seen_output_parts += 1;
            last_output = Some(*axis_ref);
        } else {
            saw_reduction = true;
        }
    }

    if seen_output_parts != output_parts {
        return Err(err("order does not contain every output loop part"));
    }

    Ok(Some(Site::At(last_output.expect(
        "reduction with output parts must see one output loop part",
    ))))
}

fn expected_axis_refs(splits: &[SplitList]) -> Vec<AxisRef> {
    let mut expected = Vec::new();
    for (axis, split_list) in splits.iter().enumerate() {
        for part in 0..=split_list.0.len() {
            expected.push(AxisRef {
                axis: Axis(axis),
                part,
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
    validate_axis(rank, axis_ref.axis)?;
    let max_part = splits[axis_ref.axis.0].0.len();
    if axis_ref.part > max_part {
        return Err(format!(
            "references nonexistent loop part {}",
            format_axis_ref(axis_ref)
        ));
    }
    Ok(())
}

fn validate_axis(rank: usize, axis: Axis) -> Result<(), String> {
    if axis.0 >= rank {
        return Err(format!("references nonexistent axis {}", axis.0));
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
    format!("{}{}", axis_ref.axis.0, "'".repeat(axis_ref.part))
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
    use super::validate_scheduled_stage;
    use crate::ir::common::Op;
    use crate::ir::stage::{
        Axis, AxisRef, Index, Schedule, ScheduledStage, Site, SplitFactor, SplitList, Stage,
    };

    #[test]
    fn accepts_pointwise_stage() {
        let stage = ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 2,
                inputs: vec![Index(vec![Axis(0), Axis(1)]), Index(vec![Axis(0)])],
                output: Index(vec![Axis(0), Axis(1)]),
            },
            schedule: Schedule {
                splits: vec![SplitList(vec![]), SplitList(vec![])],
                order: vec![
                    AxisRef {
                        axis: Axis(0),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(1),
                        part: 0,
                    },
                ],
                compute_sites: vec![
                    None,
                    Some(Site::At(AxisRef {
                        axis: Axis(0),
                        part: 0,
                    })),
                ],
                init_site: None,
            },
        };

        assert!(validate_scheduled_stage(&stage).is_ok());
    }

    #[test]
    fn accepts_reduction_stage() {
        let stage = ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 3,
                inputs: vec![Index(vec![Axis(0), Axis(1), Axis(2)])],
                output: Index(vec![Axis(0), Axis(1)]),
            },
            schedule: Schedule {
                splits: vec![
                    SplitList(vec![SplitFactor(4)]),
                    SplitList(vec![]),
                    SplitList(vec![]),
                ],
                order: vec![
                    AxisRef {
                        axis: Axis(0),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(0),
                        part: 1,
                    },
                    AxisRef {
                        axis: Axis(1),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(2),
                        part: 0,
                    },
                ],
                compute_sites: vec![None],
                init_site: Some(Site::At(AxisRef {
                    axis: Axis(1),
                    part: 0,
                })),
            },
        };

        assert!(validate_scheduled_stage(&stage).is_ok());
    }

    #[test]
    fn rejects_split_len_mismatch() {
        let error = validate_scheduled_stage(&ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 1,
                inputs: vec![Index(vec![Axis(0)])],
                output: Index(vec![Axis(0)]),
            },
            schedule: Schedule {
                splits: vec![],
                order: vec![AxisRef {
                    axis: Axis(0),
                    part: 0,
                }],
                compute_sites: vec![None],
                init_site: None,
            },
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "schedule has 0 split lists for rank 1");
    }

    #[test]
    fn rejects_missing_loop_part() {
        let error = validate_scheduled_stage(&ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 1,
                inputs: vec![Index(vec![Axis(0)])],
                output: Index(vec![Axis(0)]),
            },
            schedule: Schedule {
                splits: vec![SplitList(vec![SplitFactor(4)])],
                order: vec![AxisRef {
                    axis: Axis(0),
                    part: 0,
                }],
                compute_sites: vec![None],
                init_site: None,
            },
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "order must contain 2 loop parts, found 1"
        );
    }

    #[test]
    fn rejects_invalid_compute_site() {
        let error = validate_scheduled_stage(&ScheduledStage {
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
                compute_sites: vec![Some(Site::At(AxisRef {
                    axis: Axis(0),
                    part: 1,
                }))],
                init_site: None,
            },
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "compute site for input 0 references nonexistent loop part 0'"
        );
    }

    #[test]
    fn rejects_pointwise_init_site() {
        let error = validate_scheduled_stage(&ScheduledStage {
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
                init_site: Some(Site::Root),
            },
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "pointwise stage cannot have an init site"
        );
    }

    #[test]
    fn rejects_reduction_without_init_site() {
        let error = validate_scheduled_stage(&ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 2,
                inputs: vec![Index(vec![Axis(0), Axis(1)])],
                output: Index(vec![Axis(0)]),
            },
            schedule: Schedule {
                splits: vec![SplitList(vec![]), SplitList(vec![])],
                order: vec![
                    AxisRef {
                        axis: Axis(0),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(1),
                        part: 0,
                    },
                ],
                compute_sites: vec![None],
                init_site: None,
            },
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "reduction stage must have an init site");
    }

    #[test]
    fn rejects_interleaved_reduction_order() {
        let error = validate_scheduled_stage(&ScheduledStage {
            stage: Stage {
                op: Op::Add,
                rank: 3,
                inputs: vec![Index(vec![Axis(0), Axis(1), Axis(2)])],
                output: Index(vec![Axis(0), Axis(2)]),
            },
            schedule: Schedule {
                splits: vec![SplitList(vec![]), SplitList(vec![]), SplitList(vec![])],
                order: vec![
                    AxisRef {
                        axis: Axis(0),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(1),
                        part: 0,
                    },
                    AxisRef {
                        axis: Axis(2),
                        part: 0,
                    },
                ],
                compute_sites: vec![None],
                init_site: Some(Site::At(AxisRef {
                    axis: Axis(2),
                    part: 0,
                })),
            },
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "reduction loops cannot appear before output loops complete"
        );
    }
}

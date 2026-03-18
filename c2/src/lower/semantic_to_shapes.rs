use std::collections::BTreeMap;
use std::fmt;

use crate::ir::common::{Extent, ExtentSource, Op, Scalar, ValueId};
use crate::ir::iir::{OutputTensor, ShapeData};
use crate::ir::semantic_graph::{Graph, LocalExtentSource, Stage, Use};

pub fn lower_semantic_to_shape_data(graph: &Graph) -> Result<ShapeData, LowerError> {
    let stage_by_value = build_stage_index(graph)?;
    let mut cache = BTreeMap::new();
    let mut states = BTreeMap::new();

    let outputs = graph
        .outputs
        .iter()
        .map(|value| {
            let tensor = resolve_tensor(*value, graph, &stage_by_value, &mut cache, &mut states)?;
            Ok(OutputTensor {
                scalar: tensor.scalar,
                shape: tensor.shape,
            })
        })
        .collect::<Result<Vec<_>, LowerError>>()?;

    Ok(ShapeData { outputs })
}

fn build_stage_index(graph: &Graph) -> Result<BTreeMap<ValueId, &Stage>, LowerError> {
    let mut stage_by_value = BTreeMap::new();

    for stage in &graph.stages {
        if stage.value < graph.inputs.len() {
            return Err(LowerError::new(format!(
                "stage value {} conflicts with graph input value range 0..{}",
                stage.value,
                graph.inputs.len()
            )));
        }

        if stage_by_value.insert(stage.value, stage).is_some() {
            return Err(LowerError::new(format!(
                "duplicate stage definition for value {}",
                stage.value
            )));
        }
    }

    Ok(stage_by_value)
}

fn resolve_tensor(
    value: ValueId,
    graph: &Graph,
    stage_by_value: &BTreeMap<ValueId, &Stage>,
    cache: &mut BTreeMap<ValueId, TensorInfo>,
    states: &mut BTreeMap<ValueId, VisitState>,
) -> Result<TensorInfo, LowerError> {
    if let Some(info) = cache.get(&value) {
        return Ok(info.clone());
    }

    if value < graph.inputs.len() {
        let info = input_tensor_info(value, &graph.inputs[value]);
        cache.insert(value, info.clone());
        return Ok(info);
    }

    match states.get(&value) {
        Some(VisitState::Visiting) => {
            return Err(LowerError::new(format!(
                "cycle detected while resolving value {}",
                value
            )));
        }
        Some(VisitState::Done) => {
            return Ok(cache
                .get(&value)
                .expect("resolved value must be cached")
                .clone());
        }
        None => {}
    }

    let stage = stage_by_value
        .get(&value)
        .copied()
        .ok_or_else(|| LowerError::new(format!("unknown value {}", value)))?;

    states.insert(value, VisitState::Visiting);
    let info = stage_tensor_info(stage, graph, stage_by_value, cache, states)?;
    states.insert(value, VisitState::Done);
    cache.insert(value, info.clone());
    Ok(info)
}

fn input_tensor_info(value: ValueId, ty: &crate::ir::common::TensorType) -> TensorInfo {
    let shape = ty
        .shape
        .0
        .iter()
        .enumerate()
        .map(|(dim, extent)| match extent {
            Extent::Known(size) => ExtentSource::Const(*size),
            Extent::Param(_) => ExtentSource::InputDim { input: value, dim },
        })
        .collect();

    TensorInfo {
        scalar: ty.scalar,
        shape,
    }
}

fn stage_tensor_info(
    stage: &Stage,
    graph: &Graph,
    stage_by_value: &BTreeMap<ValueId, &Stage>,
    cache: &mut BTreeMap<ValueId, TensorInfo>,
    states: &mut BTreeMap<ValueId, VisitState>,
) -> Result<TensorInfo, LowerError> {
    let input_tensors = stage
        .inputs
        .iter()
        .map(|use_| resolve_tensor(use_.value, graph, stage_by_value, cache, states))
        .collect::<Result<Vec<_>, LowerError>>()?;

    let shape = stage
        .output
        .0
        .iter()
        .map(|axis_index| {
            let axis = stage.axes.get(*axis_index).ok_or_else(|| {
                LowerError::new(format!(
                    "stage value {} output references nonexistent axis {}",
                    stage.value, axis_index
                ))
            })?;

            resolve_extent_source(stage.value, &axis.extent, &stage.inputs, &input_tensors)
        })
        .collect::<Result<Vec<_>, LowerError>>()?;

    Ok(TensorInfo {
        scalar: scalar_for_op(stage.op),
        shape,
    })
}

fn resolve_extent_source(
    stage_value: ValueId,
    extent: &LocalExtentSource,
    inputs: &[Use],
    input_tensors: &[TensorInfo],
) -> Result<ExtentSource, LowerError> {
    match extent {
        LocalExtentSource::Const(size) => Ok(ExtentSource::Const(*size)),
        LocalExtentSource::InputDim { input, dim } => {
            let use_ = inputs.get(*input).ok_or_else(|| {
                LowerError::new(format!(
                    "stage value {} axis references nonexistent input {}",
                    stage_value, input
                ))
            })?;
            let tensor = input_tensors.get(*input).ok_or_else(|| {
                LowerError::new(format!(
                    "stage value {} missing resolved tensor for input {}",
                    stage_value, input
                ))
            })?;

            tensor.shape.get(*dim).cloned().ok_or_else(|| {
                LowerError::new(format!(
                    "stage value {} input {} (value {}) has no dimension {}",
                    stage_value, input, use_.value, dim
                ))
            })
        }
    }
}

fn scalar_for_op(op: Op) -> Scalar {
    match op {
        Op::Add | Op::Mul | Op::Div | Op::Sub | Op::Max | Op::Min | Op::Pow | Op::Log => {
            Scalar::Float
        }
        Op::Gt
        | Op::Ge
        | Op::Lt
        | Op::Le
        | Op::Eq
        | Op::Ne
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Not => Scalar::Int,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorInfo {
    scalar: Scalar,
    shape: Vec<ExtentSource>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisitState {
    Visiting,
    Done,
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
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LowerError {}

#[cfg(test)]
mod tests {
    use super::lower_semantic_to_shape_data;
    use crate::ir::common::{Extent, ExtentSource, Op, Scalar, Shape, TensorType};
    use crate::ir::iir::{OutputTensor, ShapeData};
    use crate::ir::semantic_graph::{Axis, Graph, Index, LocalExtentSource, Stage, Use};

    #[test]
    fn lowers_direct_input_outputs_to_global_sources() {
        let graph = Graph {
            inputs: vec![
                TensorType {
                    scalar: Scalar::Float,
                    shape: Shape(vec![Extent::Param("n".to_string()), Extent::Known(4)]),
                },
                TensorType {
                    scalar: Scalar::Int,
                    shape: Shape(vec![Extent::Param("m".to_string())]),
                },
            ],
            stages: vec![],
            outputs: vec![0, 1],
        };

        let shapes = lower_semantic_to_shape_data(&graph).unwrap();

        assert_eq!(
            shapes,
            ShapeData {
                outputs: vec![
                    OutputTensor {
                        scalar: Scalar::Float,
                        shape: vec![
                            ExtentSource::InputDim { input: 0, dim: 0 },
                            ExtentSource::Const(4),
                        ],
                    },
                    OutputTensor {
                        scalar: Scalar::Int,
                        shape: vec![ExtentSource::InputDim { input: 1, dim: 0 }],
                    },
                ],
            }
        );
    }

    #[test]
    fn lowers_stage_output_shapes_through_producer_chain() {
        let graph = Graph {
            inputs: vec![float_input(2)],
            stages: vec![
                Stage {
                    value: 1,
                    expr: 0,
                    op: Op::Add,
                    inputs: vec![Use {
                        value: 0,
                        index: Index(vec![0, 1]),
                    }],
                    axes: vec![axis(input_dim(0, 0)), axis(input_dim(0, 1))],
                    output: Index(vec![0]),
                },
                Stage {
                    value: 2,
                    expr: 1,
                    op: Op::Div,
                    inputs: vec![Use {
                        value: 1,
                        index: Index(vec![0]),
                    }],
                    axes: vec![axis(input_dim(0, 0))],
                    output: Index(vec![0]),
                },
            ],
            outputs: vec![2],
        };

        let shapes = lower_semantic_to_shape_data(&graph).unwrap();

        assert_eq!(
            shapes.outputs,
            vec![OutputTensor {
                scalar: Scalar::Float,
                shape: vec![ExtentSource::InputDim { input: 0, dim: 0 }],
            }]
        );
    }

    #[test]
    fn lowers_boolean_ops_to_int_outputs() {
        let graph = Graph {
            inputs: vec![float_input(1), float_input(1)],
            stages: vec![Stage {
                value: 2,
                expr: 0,
                op: Op::Gt,
                inputs: vec![
                    Use {
                        value: 0,
                        index: Index(vec![0]),
                    },
                    Use {
                        value: 1,
                        index: Index(vec![0]),
                    },
                ],
                axes: vec![axis(input_dim(0, 0))],
                output: Index(vec![0]),
            }],
            outputs: vec![2],
        };

        let shapes = lower_semantic_to_shape_data(&graph).unwrap();

        assert_eq!(shapes.outputs[0].scalar, Scalar::Int);
        assert_eq!(
            shapes.outputs[0].shape,
            vec![ExtentSource::InputDim { input: 0, dim: 0 }]
        );
    }

    #[test]
    fn lowers_constant_local_extent_sources() {
        let graph = Graph {
            inputs: vec![float_input(1)],
            stages: vec![Stage {
                value: 1,
                expr: 0,
                op: Op::Add,
                inputs: vec![Use {
                    value: 0,
                    index: Index(vec![0]),
                }],
                axes: vec![axis(LocalExtentSource::Const(8))],
                output: Index(vec![0]),
            }],
            outputs: vec![1],
        };

        let shapes = lower_semantic_to_shape_data(&graph).unwrap();

        assert_eq!(shapes.outputs[0].shape, vec![ExtentSource::Const(8)]);
    }

    fn float_input(rank: usize) -> TensorType {
        TensorType {
            scalar: Scalar::Float,
            shape: Shape(
                (0..rank)
                    .map(|dim| Extent::Param(format!("input_dim{dim}")))
                    .collect(),
            ),
        }
    }

    fn input_dim(input: usize, dim: usize) -> LocalExtentSource {
        LocalExtentSource::InputDim { input, dim }
    }

    fn axis(extent: LocalExtentSource) -> Axis {
        Axis { extent }
    }
}

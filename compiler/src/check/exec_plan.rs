use std::collections::BTreeSet;
use std::fmt;

use crate::ir::common::DimRef;
use crate::ir::exec_plan::{Arg, BufferRef, ExecPlan, Input, Intermediate, KernelId, Param, Step};
use crate::ir::kernel_program::{Access, Action, Block, Iter, Kernel, LoopId, ScalarExpr};

pub fn validate_exec_plan(plan: &ExecPlan) -> Result<(), ValidationError> {
    validate_metadata(plan)?;
    validate_buffers(plan)?;
    validate_kernels(plan)?;
    validate_exec(plan)
}

fn validate_metadata(plan: &ExecPlan) -> Result<(), ValidationError> {
    if plan.count != plan.ranks.len() {
        return Err(err(format!(
            "count is {} but ranks has {} entries",
            plan.count,
            plan.ranks.len()
        )));
    }
    if plan.count != plan.shapes.len() {
        return Err(err(format!(
            "count is {} but shapes has {} entries",
            plan.count,
            plan.shapes.len()
        )));
    }
    if plan.count != plan.buffers.outputs.len() {
        return Err(err(format!(
            "count is {} but outputs has {} entries",
            plan.count,
            plan.buffers.outputs.len()
        )));
    }

    for (output, rank) in plan.ranks.iter().enumerate() {
        if *rank != plan.shapes[output].0.len() {
            return Err(err(format!(
                "output {} rank is {} but shape has {} dims",
                output,
                rank,
                plan.shapes[output].0.len()
            )));
        }
    }

    Ok(())
}

fn validate_buffers(plan: &ExecPlan) -> Result<(), ValidationError> {
    for (input, buffer) in plan.buffers.inputs.iter().enumerate() {
        validate_shape(plan, &buffer.shape)
            .map_err(|message| err(format!("input {} shape {}", input, message)))?;
        validate_layout(plan, &buffer.layout)
            .map_err(|message| err(format!("input {} layout {}", input, message)))?;
    }
    for (intermediate, buffer) in plan.buffers.intermediates.iter().enumerate() {
        validate_shape(plan, &buffer.shape)
            .map_err(|message| err(format!("intermediate {} shape {}", intermediate, message)))?;
        validate_layout(plan, &buffer.layout)
            .map_err(|message| err(format!("intermediate {} layout {}", intermediate, message)))?;
    }
    for (output, buffer) in plan.buffers.outputs.iter().enumerate() {
        validate_shape(plan, &buffer.shape)
            .map_err(|message| err(format!("output {} shape {}", output, message)))?;
        validate_layout(plan, &buffer.layout)
            .map_err(|message| err(format!("output {} layout {}", output, message)))?;
        if plan.shapes[output] != buffer.shape {
            return Err(err(format!(
                "output {} public shape differs from buffer shape",
                output
            )));
        }
    }
    Ok(())
}

fn validate_shape(plan: &ExecPlan, shape: &crate::ir::exec_plan::Shape) -> Result<(), String> {
    for dim in &shape.0 {
        validate_input_dim(plan, dim.buffer, dim.dim)?;
    }
    Ok(())
}

fn validate_layout(plan: &ExecPlan, layout: &crate::ir::exec_plan::Layout) -> Result<(), String> {
    for extent in &layout.0 {
        validate_input_dim(plan, extent.source.buffer, extent.source.dim)?;
    }
    Ok(())
}

fn validate_kernels(plan: &ExecPlan) -> Result<(), ValidationError> {
    for (kernel_index, kernel) in plan.kernels.iter().enumerate() {
        validate_kernel_params(kernel)
            .map_err(|message| err(format!("kernel {} {}", kernel_index, message)))?;
        let mut loops = BTreeSet::new();
        collect_loop_ids(&kernel.body, &mut loops)
            .map_err(|error| err(format!("kernel {} {}", kernel_index, error.message)))?;
    }
    Ok(())
}

fn validate_kernel_params(kernel: &Kernel<Param>) -> Result<(), String> {
    for (index, param) in kernel.reads.iter().enumerate() {
        if *param
            != (Param {
                arg: Arg::Readonly,
                ind: index,
            })
        {
            return Err(format!("read {} is {:?}", index, param));
        }
    }
    for (index, param) in kernel.writes.iter().enumerate() {
        if *param
            != (Param {
                arg: Arg::Writeable,
                ind: index,
            })
        {
            return Err(format!("write {} is {:?}", index, param));
        }
    }
    Ok(())
}

fn validate_exec(plan: &ExecPlan) -> Result<(), ValidationError> {
    let mut allocated = BTreeSet::new();
    let mut dispatched = BTreeSet::new();

    for (step_index, step) in plan.exec.0.iter().enumerate() {
        match step {
            Step::Alloc(intermediate) => {
                validate_intermediate(plan, *intermediate)
                    .map_err(|message| err(format!("step {} alloc {}", step_index, message)))?;
                if !allocated.insert(intermediate.0) {
                    return Err(err(format!(
                        "step {} alloc repeats intermediate {}",
                        step_index, intermediate.0
                    )));
                }
            }
            Step::Dispatch {
                kernel,
                reads,
                writes,
            } => {
                validate_kernel_id(plan, *kernel)
                    .map_err(|message| err(format!("step {} dispatch {}", step_index, message)))?;
                if !dispatched.insert(kernel.0) {
                    return Err(err(format!(
                        "step {} dispatch repeats kernel {}",
                        step_index, kernel.0
                    )));
                }
                validate_dispatch(plan, &allocated, *kernel, reads, writes)
                    .map_err(|message| err(format!("step {} dispatch {}", step_index, message)))?;
            }
            Step::Free(intermediate) => {
                validate_intermediate(plan, *intermediate)
                    .map_err(|message| err(format!("step {} free {}", step_index, message)))?;
                if !allocated.remove(&intermediate.0) {
                    return Err(err(format!(
                        "step {} free intermediate {} is not allocated",
                        step_index, intermediate.0
                    )));
                }
            }
        }
    }

    for kernel in 0..plan.kernels.len() {
        if !dispatched.contains(&kernel) {
            return Err(err(format!("kernel {} is not dispatched", kernel)));
        }
    }
    if let Some(intermediate) = allocated.iter().next() {
        return Err(err(format!(
            "intermediate {} remains allocated",
            intermediate
        )));
    }

    Ok(())
}

fn validate_dispatch(
    plan: &ExecPlan,
    allocated: &BTreeSet<usize>,
    kernel: KernelId,
    reads: &[BufferRef],
    writes: &[BufferRef],
) -> Result<(), String> {
    let body = &plan.kernels[kernel.0];
    if reads.len() != body.reads.len() {
        return Err(format!(
            "kernel {} has {} reads but dispatch binds {}",
            kernel.0,
            body.reads.len(),
            reads.len()
        ));
    }
    if writes.len() != body.writes.len() {
        return Err(format!(
            "kernel {} has {} writes but dispatch binds {}",
            kernel.0,
            body.writes.len(),
            writes.len()
        ));
    }

    for buffer in reads.iter().chain(writes) {
        validate_buffer_ref(plan, *buffer)?;
        validate_live_buffer(allocated, *buffer)?;
    }

    let mut loops = BTreeSet::new();
    collect_loop_ids(&body.body, &mut loops).map_err(|error| error.message)?;
    validate_block(plan, reads, writes, &loops, &BTreeSet::new(), &body.body)
}

fn collect_loop_ids(
    block: &Block<Param>,
    loops: &mut BTreeSet<usize>,
) -> Result<(), ValidationError> {
    for action in &block.0 {
        if let Action::Loop { id, body, .. } = action {
            if !loops.insert(id.0) {
                return Err(err(format!("loop id {} is repeated", id.0)));
            }
            collect_loop_ids(body, loops)?;
        }
    }
    Ok(())
}

fn validate_block(
    plan: &ExecPlan,
    reads: &[BufferRef],
    writes: &[BufferRef],
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    block: &Block<Param>,
) -> Result<(), String> {
    for action in &block.0 {
        match action {
            Action::Loop {
                id, extent, body, ..
            } => {
                validate_param_dim(plan, reads, writes, extent.source)
                    .map_err(|message| format!("loop {} extent {}", id.0, message))?;
                let mut child_scope = scope.clone();
                child_scope.insert(id.0);
                validate_block(plan, reads, writes, loops, &child_scope, body)?;
            }
            Action::Init {
                write, zero_checks, ..
            } => {
                validate_write_access(plan, reads, writes, loops, scope, write)
                    .map_err(|message| format!("init {}", message))?;
                for loop_id in zero_checks {
                    validate_loop_ref(loops, scope, *loop_id)
                        .map_err(|message| format!("init zero check {}", message))?;
                }
            }
            Action::Compute {
                write, reads: ins, ..
            } => {
                validate_write_access(plan, reads, writes, loops, scope, write)
                    .map_err(|message| format!("compute write {}", message))?;
                for (read_index, read) in ins.iter().enumerate() {
                    validate_access(plan, reads, writes, loops, scope, read)
                        .map_err(|message| format!("compute read {} {}", read_index, message))?;
                }
            }
            Action::Snapshot { write, read } => {
                validate_write_access(plan, reads, writes, loops, scope, write)
                    .map_err(|message| format!("snapshot write {}", message))?;
                validate_access(plan, reads, writes, loops, scope, read)
                    .map_err(|message| format!("snapshot read {}", message))?;
            }
            Action::Scale { write, factor } => {
                validate_write_access(plan, reads, writes, loops, scope, write)
                    .map_err(|message| format!("scale write {}", message))?;
                validate_scale_expr(plan, reads, writes, loops, scope, factor)
                    .map_err(|message| format!("scale factor {}", message))?;
            }
        }
    }
    Ok(())
}

fn validate_scale_expr(
    plan: &ExecPlan,
    reads: &[BufferRef],
    writes: &[BufferRef],
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    expr: &ScalarExpr<Param>,
) -> Result<(), String> {
    match expr {
        ScalarExpr::Access(access) => validate_access(plan, reads, writes, loops, scope, access),
        ScalarExpr::Unary { arg, .. } => {
            validate_scale_expr(plan, reads, writes, loops, scope, arg)
        }
        ScalarExpr::Binary { lhs, rhs, .. } => {
            validate_scale_expr(plan, reads, writes, loops, scope, lhs)?;
            validate_scale_expr(plan, reads, writes, loops, scope, rhs)
        }
    }
}

fn validate_write_access(
    plan: &ExecPlan,
    reads: &[BufferRef],
    writes: &[BufferRef],
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    access: &Access<Param>,
) -> Result<(), String> {
    validate_param(access.buffer, reads, writes)?;
    if !matches!(access.buffer.arg, Arg::Writeable) {
        return Err(format!("writes {:?} outside writeables", access.buffer));
    }
    validate_access(plan, reads, writes, loops, scope, access)
}

fn validate_access(
    plan: &ExecPlan,
    reads: &[BufferRef],
    writes: &[BufferRef],
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    access: &Access<Param>,
) -> Result<(), String> {
    let buffer = resolve_param(access.buffer, reads, writes)?;
    let layout_rank = buffer_layout_rank(plan, buffer)?;
    if access.index.len() != layout_rank {
        return Err(format!(
            "access has {} indexes for layout rank {}",
            access.index.len(),
            layout_rank
        ));
    }
    for iter in &access.index {
        validate_iter(loops, scope, iter)?;
    }
    Ok(())
}

fn validate_iter(
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    iter: &Iter,
) -> Result<(), String> {
    match iter {
        Iter::Raw(loop_id) => validate_loop_ref(loops, scope, *loop_id),
        Iter::Reconstructed {
            loops: ids,
            factors,
        } => {
            if ids.is_empty() {
                return Err("reconstructed iterator has no loops".to_string());
            }
            if factors.len().saturating_add(1) != ids.len() {
                return Err(format!(
                    "reconstructed iterator has {} loops and {} factors",
                    ids.len(),
                    factors.len()
                ));
            }
            for loop_id in ids {
                validate_loop_ref(loops, scope, *loop_id)?;
            }
            Ok(())
        }
    }
}

fn validate_loop_ref(
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    loop_id: LoopId,
) -> Result<(), String> {
    if !loops.contains(&loop_id.0) {
        return Err(format!("references nonexistent loop {}", loop_id.0));
    }
    if !scope.contains(&loop_id.0) {
        return Err(format!("references loop {} outside scope", loop_id.0));
    }
    Ok(())
}

fn validate_param_dim(
    plan: &ExecPlan,
    reads: &[BufferRef],
    writes: &[BufferRef],
    dim: DimRef<Param>,
) -> Result<(), String> {
    let buffer = resolve_param(dim.buffer, reads, writes)?;
    let shape_rank = buffer_shape_rank(plan, buffer)?;
    if dim.dim >= shape_rank {
        return Err(format!(
            "references nonexistent dim {} of {:?}",
            dim.dim, dim.buffer
        ));
    }
    Ok(())
}

fn validate_param(param: Param, reads: &[BufferRef], writes: &[BufferRef]) -> Result<(), String> {
    resolve_param(param, reads, writes).map(|_| ())
}

fn resolve_param(
    param: Param,
    reads: &[BufferRef],
    writes: &[BufferRef],
) -> Result<BufferRef, String> {
    match param.arg {
        Arg::Readonly => reads
            .get(param.ind)
            .copied()
            .ok_or_else(|| format!("references nonexistent readonly {}", param.ind)),
        Arg::Writeable => writes
            .get(param.ind)
            .copied()
            .ok_or_else(|| format!("references nonexistent writeable {}", param.ind)),
    }
}

fn validate_input_dim(plan: &ExecPlan, input: Input, dim: usize) -> Result<(), String> {
    let rank = plan
        .buffers
        .inputs
        .get(input.0)
        .map(|buffer| buffer.shape.0.len())
        .ok_or_else(|| format!("references nonexistent input {}", input.0))?;
    if dim >= rank {
        return Err(format!(
            "references nonexistent dim {} of input {}",
            dim, input.0
        ));
    }
    Ok(())
}

fn validate_buffer_ref(plan: &ExecPlan, buffer: BufferRef) -> Result<(), String> {
    match buffer {
        BufferRef::Input(input) => {
            if input.0 >= plan.buffers.inputs.len() {
                return Err(format!("references nonexistent input {}", input.0));
            }
        }
        BufferRef::Intermediate(intermediate) => validate_intermediate(plan, intermediate)?,
        BufferRef::Output(output) => {
            if output.0 >= plan.buffers.outputs.len() {
                return Err(format!("references nonexistent output {}", output.0));
            }
        }
    }
    Ok(())
}

fn validate_live_buffer(allocated: &BTreeSet<usize>, buffer: BufferRef) -> Result<(), String> {
    if let BufferRef::Intermediate(intermediate) = buffer {
        if !allocated.contains(&intermediate.0) {
            return Err(format!(
                "references unallocated intermediate {}",
                intermediate.0
            ));
        }
    }
    Ok(())
}

fn validate_intermediate(plan: &ExecPlan, intermediate: Intermediate) -> Result<(), String> {
    if intermediate.0 >= plan.buffers.intermediates.len() {
        return Err(format!(
            "references nonexistent intermediate {}",
            intermediate.0
        ));
    }
    Ok(())
}

fn validate_kernel_id(plan: &ExecPlan, kernel: KernelId) -> Result<(), String> {
    if kernel.0 >= plan.kernels.len() {
        return Err(format!("references nonexistent kernel {}", kernel.0));
    }
    Ok(())
}

fn buffer_shape_rank(plan: &ExecPlan, buffer: BufferRef) -> Result<usize, String> {
    match buffer {
        BufferRef::Input(input) => plan
            .buffers
            .inputs
            .get(input.0)
            .map(|buffer| buffer.shape.0.len())
            .ok_or_else(|| format!("references nonexistent input {}", input.0)),
        BufferRef::Intermediate(intermediate) => plan
            .buffers
            .intermediates
            .get(intermediate.0)
            .map(|buffer| buffer.shape.0.len())
            .ok_or_else(|| format!("references nonexistent intermediate {}", intermediate.0)),
        BufferRef::Output(output) => plan
            .buffers
            .outputs
            .get(output.0)
            .map(|buffer| buffer.shape.0.len())
            .ok_or_else(|| format!("references nonexistent output {}", output.0)),
    }
}

fn buffer_layout_rank(plan: &ExecPlan, buffer: BufferRef) -> Result<usize, String> {
    match buffer {
        BufferRef::Input(input) => plan
            .buffers
            .inputs
            .get(input.0)
            .map(|buffer| buffer.layout.0.len())
            .ok_or_else(|| format!("references nonexistent input {}", input.0)),
        BufferRef::Intermediate(intermediate) => plan
            .buffers
            .intermediates
            .get(intermediate.0)
            .map(|buffer| buffer.layout.0.len())
            .ok_or_else(|| format!("references nonexistent intermediate {}", intermediate.0)),
        BufferRef::Output(output) => plan
            .buffers
            .outputs
            .get(output.0)
            .map(|buffer| buffer.layout.0.len())
            .ok_or_else(|| format!("references nonexistent output {}", output.0)),
    }
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
    use super::validate_exec_plan;
    use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
    use crate::ir::exec_plan::{
        Arg, BufferRef, Buffers, Exec, ExecPlan, Input, InputBuffer, Intermediate,
        IntermediateBuffer, KernelId, Layout, Output, OutputBuffer, Param, Shape, Step,
    };
    use crate::ir::kernel_program::{
        Access, Action, Block, Iter, Kernel, LoopId, LoopMode, TailGuard,
    };

    fn dim(input: usize, dim: usize) -> DimRef<Input> {
        DimRef {
            buffer: Input(input),
            dim,
        }
    }

    fn extent(input: usize, index: usize) -> Extent<Input> {
        Extent {
            source: dim(input, index),
            kind: ExtentKind::Semantic,
        }
    }

    fn plan() -> ExecPlan {
        ExecPlan {
            kernels: vec![Kernel {
                reads: vec![Param {
                    arg: Arg::Readonly,
                    ind: 0,
                }],
                writes: vec![Param {
                    arg: Arg::Writeable,
                    ind: 0,
                }],
                body: Block(vec![Action::Loop {
                    id: LoopId(0),
                    mode: LoopMode::Serial,
                    extent: Extent {
                        source: DimRef {
                            buffer: Param {
                                arg: Arg::Readonly,
                                ind: 0,
                            },
                            dim: 0,
                        },
                        kind: ExtentKind::Semantic,
                    },
                    guard: TailGuard(false),
                    body: Block(vec![Action::Compute {
                        op: Op::Add,
                        write: Access {
                            buffer: Param {
                                arg: Arg::Writeable,
                                ind: 0,
                            },
                            index: vec![Iter::Raw(LoopId(0))],
                        },
                        reads: vec![Access {
                            buffer: Param {
                                arg: Arg::Readonly,
                                ind: 0,
                            },
                            index: vec![Iter::Raw(LoopId(0))],
                        }],
                    }]),
                }]),
            }],
            buffers: Buffers {
                inputs: vec![InputBuffer {
                    shape: Shape(vec![dim(0, 0)]),
                    layout: Layout(vec![extent(0, 0)]),
                }],
                intermediates: vec![],
                outputs: vec![OutputBuffer {
                    shape: Shape(vec![dim(0, 0)]),
                    layout: Layout(vec![extent(0, 0)]),
                }],
            },
            count: 1,
            ranks: vec![1],
            shapes: vec![Shape(vec![dim(0, 0)])],
            exec: Exec(vec![Step::Dispatch {
                kernel: KernelId(0),
                reads: vec![BufferRef::Input(Input(0))],
                writes: vec![BufferRef::Output(Output(0))],
            }]),
        }
    }

    #[test]
    fn accepts_exec_plan() {
        assert!(validate_exec_plan(&plan()).is_ok());
    }

    #[test]
    fn rejects_count_rank_mismatch() {
        let mut plan = plan();
        plan.ranks = vec![];

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(error.to_string(), "count is 1 but ranks has 0 entries");
    }

    #[test]
    fn rejects_invalid_output_shape_dim() {
        let mut plan = plan();
        plan.buffers.outputs[0].shape = Shape(vec![dim(0, 1)]);
        plan.shapes[0] = Shape(vec![dim(0, 1)]);

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "output 0 shape references nonexistent dim 1 of input 0"
        );
    }

    #[test]
    fn rejects_bad_read_param_bucket() {
        let mut plan = plan();
        plan.kernels[0].reads[0] = Param {
            arg: Arg::Writeable,
            ind: 0,
        };

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "kernel 0 read 0 is Param { arg: Writeable, ind: 0 }"
        );
    }

    #[test]
    fn rejects_dispatch_read_arity_mismatch() {
        let mut plan = plan();
        let Step::Dispatch { reads, .. } = &mut plan.exec.0[0] else {
            unreachable!();
        };
        reads.clear();

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "step 0 dispatch kernel 0 has 1 reads but dispatch binds 0"
        );
    }

    #[test]
    fn rejects_unallocated_intermediate_dispatch() {
        let mut plan = plan();
        plan.buffers.intermediates.push(IntermediateBuffer {
            shape: Shape(vec![dim(0, 0)]),
            layout: Layout(vec![extent(0, 0)]),
        });
        let Step::Dispatch { reads, .. } = &mut plan.exec.0[0] else {
            unreachable!();
        };
        reads[0] = BufferRef::Intermediate(Intermediate(0));

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "step 0 dispatch references unallocated intermediate 0"
        );
    }

    #[test]
    fn rejects_access_rank_mismatch() {
        let mut plan = plan();
        let Action::Loop { body, .. } = &mut plan.kernels[0].body.0[0] else {
            unreachable!();
        };
        let Action::Compute { write, .. } = &mut body.0[0] else {
            unreachable!();
        };
        write.index = vec![];

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "step 0 dispatch compute write access has 0 indexes for layout rank 1"
        );
    }

    #[test]
    fn rejects_loop_outside_scope() {
        let mut plan = plan();
        let Action::Loop { body, .. } = &mut plan.kernels[0].body.0[0] else {
            unreachable!();
        };
        let Action::Compute { write, .. } = &mut body.0[0] else {
            unreachable!();
        };
        write.index = vec![Iter::Raw(LoopId(1))];

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(
            error.to_string(),
            "step 0 dispatch compute write references nonexistent loop 1"
        );
    }

    #[test]
    fn rejects_unfreed_intermediate() {
        let mut plan = plan();
        plan.buffers.intermediates.push(IntermediateBuffer {
            shape: Shape(vec![dim(0, 0)]),
            layout: Layout(vec![extent(0, 0)]),
        });
        plan.exec.0.insert(0, Step::Alloc(Intermediate(0)));

        let error = validate_exec_plan(&plan).unwrap_err();
        assert_eq!(error.to_string(), "intermediate 0 remains allocated");
    }
}

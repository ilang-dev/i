use std::fmt;

use crate::check::exec_plan::validate_exec_plan;
use crate::check::parallel_module::validate_parallel_module;
use crate::ir::exec_plan::{BoundKernel, ExecPlan, KernelId, Step};
use crate::ir::parallel_module::{
    DeviceKernel, DeviceKernelId, HostExec, HostStep, ParallelModule,
};

pub fn lower_exec_plan_to_parallel_module(plan: &ExecPlan) -> Result<ParallelModule, LowerError> {
    validate_exec_plan(plan).map_err(LowerError::from_exec_plan)?;

    let module = ParallelModule {
        kernels: plan.kernels.iter().map(lower_kernel).collect(),
        buffers: plan.buffers.clone(),
        count: plan.count,
        ranks: plan.ranks.clone(),
        shapes: plan.shapes.clone(),
        exec: HostExec(plan.exec.0.iter().map(lower_step).collect()),
    };

    validate_parallel_module(&module).map_err(LowerError::from_parallel_module)?;
    Ok(module)
}

fn lower_kernel(kernel: &BoundKernel) -> DeviceKernel {
    DeviceKernel {
        reads: kernel.reads.clone(),
        writes: kernel.writes.clone(),
        locals: kernel.locals.clone(),
        execution: kernel.execution.clone(),
        body: kernel.body.clone(),
    }
}

fn lower_step(step: &Step) -> HostStep {
    match step {
        Step::Alloc(intermediate) => HostStep::Alloc(*intermediate),
        Step::Dispatch {
            kernel,
            reads,
            writes,
        } => HostStep::Launch {
            kernel: lower_kernel_id(*kernel),
            reads: reads.clone(),
            writes: writes.clone(),
        },
        Step::Free(intermediate) => HostStep::Free(*intermediate),
    }
}

fn lower_kernel_id(kernel: KernelId) -> DeviceKernelId {
    DeviceKernelId(kernel.0)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LowerError {
    pub message: String,
}

impl LowerError {
    fn from_exec_plan(error: crate::check::exec_plan::ValidationError) -> Self {
        Self {
            message: error.to_string(),
        }
    }

    fn from_parallel_module(error: crate::check::parallel_module::ValidationError) -> Self {
        Self {
            message: error.to_string(),
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
    use super::lower_exec_plan_to_parallel_module;
    use crate::front::parse_expr;
    use crate::ir::parallel_module::{DeviceKernelId, HostStep};
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::kernel_program_to_exec_plan::lower_kernel_program_to_exec_plan;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;
    use crate::{component, front};

    fn lower_expr(src: &str) -> crate::ir::parallel_module::ParallelModule {
        let component = component::expr(parse_expr(src).unwrap());
        lower_component(&component)
    }

    fn lower_component(
        component: &crate::ir::component::Component,
    ) -> crate::ir::parallel_module::ParallelModule {
        let graph = lower_component_to_graph(component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        lower_exec_plan_to_parallel_module(&exec_plan).unwrap()
    }

    #[test]
    fn lowers_dispatch_to_host_launch() {
        let module = lower_expr("i+i~i");

        assert!(module.exec.0.iter().any(|step| {
            matches!(step, HostStep::Launch { kernel, reads, writes }
                if *kernel == DeviceKernelId(0) && reads.len() == 2 && writes.len() == 1)
        }));
    }

    #[test]
    fn preserves_split_execution_shape() {
        let module = lower_expr("ijk+ijk~ijk|i:16,j:16,k:4|ii'jj'kk'");

        assert_eq!(module.kernels[0].execution.groups.len(), 3);
        assert_eq!(module.kernels[0].execution.lanes.len(), 3);
        assert_eq!(
            module.kernels[0]
                .execution
                .groups
                .iter()
                .map(|dim| dim.id.0)
                .collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
        assert_eq!(
            module.kernels[0]
                .execution
                .lanes
                .iter()
                .map(|dim| dim.id.0)
                .collect::<Vec<_>>(),
            vec![1, 3, 5]
        );
    }

    #[test]
    fn preserves_host_allocation_steps() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let module = lower_component(&component);

        assert!(matches!(module.exec.0[0], HostStep::Alloc(_)));
        assert!(matches!(module.exec.0[1], HostStep::Launch { .. }));
        assert!(matches!(module.exec.0[2], HostStep::Launch { .. }));
        assert!(matches!(module.exec.0[3], HostStep::Free(_)));
    }
}

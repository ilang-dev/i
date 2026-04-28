use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::exec_plan::validate_exec_plan;
use crate::check::module::validate_module;
use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
use crate::ir::exec_plan::{Arg, BufferRef, ExecPlan, Input, Param, Step};
use crate::ir::kernel_program::{Access, Action, Block as KernelBlock, Iter, Kernel, LoopId};
use crate::ir::module::{
    Block, Cast, Expr, Field, Fn, Ident, Module, Place, Signature, Stmt, Type,
};

pub fn lower_exec_plan_to_module(plan: &ExecPlan) -> Result<Module, LowerError> {
    validate_exec_plan(plan).map_err(LowerError::from_exec_plan)?;
    let module = Builder { plan }.lower()?;
    validate_module(&module).map_err(LowerError::from_module)?;
    Ok(module)
}

struct Builder<'a> {
    plan: &'a ExecPlan,
}

impl<'a> Builder<'a> {
    fn lower(&self) -> Result<Module, LowerError> {
        Ok(Module {
            count: self.lower_count(),
            ranks: self.lower_ranks(),
            shapes: self.lower_shapes(),
            exec: self.lower_exec()?,
            kernels: self.lower_kernels()?,
        })
    }

    fn lower_count(&self) -> Fn {
        Fn {
            ident: id("count"),
            signature: Signature::Count,
            body: Block(vec![Stmt::Return(Some(Expr::Usize(self.plan.count)))]),
        }
    }

    fn lower_ranks(&self) -> Fn {
        Fn {
            ident: id("ranks"),
            signature: Signature::Ranks,
            body: Block(
                self.plan
                    .ranks
                    .iter()
                    .enumerate()
                    .map(|(output, rank)| Stmt::Set {
                        dst: index_place(Place::Ident(id("ranks")), Expr::Usize(output)),
                        value: Expr::Usize(*rank),
                    })
                    .chain(std::iter::once(Stmt::Return(None)))
                    .collect(),
            ),
        }
    }

    fn lower_shapes(&self) -> Fn {
        let mut statements = Vec::new();
        for (output, shape) in self.plan.shapes.iter().enumerate() {
            for (dim, source) in shape.0.iter().enumerate() {
                statements.push(Stmt::Set {
                    dst: index_place(
                        index_place(Place::Ident(id("shapes")), Expr::Usize(output)),
                        Expr::Usize(dim),
                    ),
                    value: self.input_dim(*source),
                });
            }
        }
        statements.push(Stmt::Return(None));

        Fn {
            ident: id("shapes"),
            signature: Signature::Shapes,
            body: Block(statements),
        }
    }

    fn lower_exec(&self) -> Result<Fn, LowerError> {
        let mut statements = Vec::new();

        for input in 0..self.plan.buffers.inputs.len() {
            statements.push(Stmt::Let {
                ident: input_ident(Input(input)),
                ty: Type::View,
                value: Expr::Cast {
                    kind: Cast::View,
                    value: Box::new(index_expr(Expr::Ident(id("inputs")), Expr::Usize(input))),
                },
            });
        }
        for output in 0..self.plan.buffers.outputs.len() {
            statements.push(Stmt::Let {
                ident: output_ident(crate::ir::exec_plan::Output(output)),
                ty: Type::ViewMut,
                value: Expr::Cast {
                    kind: Cast::ViewMut,
                    value: Box::new(index_expr(Expr::Ident(id("outputs")), Expr::Usize(output))),
                },
            });
        }

        for step in &self.plan.exec.0 {
            match step {
                Step::Alloc(intermediate) => {
                    let buffer = &self.plan.buffers.intermediates[intermediate.0];
                    statements.push(Stmt::Alloc {
                        dst: intermediate_ident(*intermediate),
                        shape: buffer
                            .shape
                            .0
                            .iter()
                            .copied()
                            .map(|dim| self.input_dim(dim))
                            .collect(),
                        layout: buffer
                            .layout
                            .0
                            .iter()
                            .map(|extent| self.input_extent(extent))
                            .collect(),
                    });
                }
                Step::Dispatch {
                    kernel,
                    reads,
                    writes,
                } => {
                    statements.push(Stmt::Dispatch {
                        kernel: kernel_ident(kernel.0),
                        reads: reads
                            .iter()
                            .copied()
                            .map(Self::buffer_ident)
                            .collect::<Result<Vec<_>, _>>()?,
                        writes: writes
                            .iter()
                            .copied()
                            .map(Self::buffer_ident)
                            .collect::<Result<Vec<_>, _>>()?,
                    });
                }
                Step::Free(intermediate) => {
                    statements.push(Stmt::Free(intermediate_ident(*intermediate)));
                }
            }
        }
        statements.push(Stmt::Return(None));

        Ok(Fn {
            ident: id("exec"),
            signature: Signature::Exec,
            body: Block(statements),
        })
    }

    fn lower_kernels(&self) -> Result<Vec<Fn>, LowerError> {
        self.plan
            .kernels
            .iter()
            .enumerate()
            .map(|(kernel_index, kernel)| self.lower_kernel(kernel_index, kernel))
            .collect()
    }

    fn lower_kernel(&self, kernel_index: usize, kernel: &Kernel<Param>) -> Result<Fn, LowerError> {
        let initialized = initialized_buffers(&kernel.body);
        let mut lower = KernelLowerer {
            initialized,
            loops: BTreeMap::new(),
        };
        let body = lower.lower_block(&kernel.body)?;
        Ok(Fn {
            ident: kernel_ident(kernel_index),
            signature: Signature::Kernel,
            body,
        })
    }

    fn input_dim(&self, dim: DimRef<Input>) -> Expr {
        index_expr(
            field_expr(
                index_expr(Expr::Ident(id("inputs")), Expr::Usize(dim.buffer.0)),
                Field::Shape,
            ),
            Expr::Usize(dim.dim),
        )
    }

    fn input_extent(&self, extent: &Extent<Input>) -> Expr {
        extent_expr(self.input_dim(extent.source), &extent.kind)
    }

    fn buffer_ident(buffer: BufferRef) -> Result<Ident, LowerError> {
        match buffer {
            BufferRef::Input(input) => Ok(input_ident(input)),
            BufferRef::Intermediate(intermediate) => Ok(intermediate_ident(intermediate)),
            BufferRef::Output(output) => Ok(output_ident(output)),
        }
    }
}

struct KernelLowerer {
    initialized: BTreeSet<Param>,
    loops: BTreeMap<LoopId, LoopInfo>,
}

impl KernelLowerer {
    fn lower_block(&mut self, block: &KernelBlock<Param>) -> Result<Block, LowerError> {
        block
            .0
            .iter()
            .map(|action| self.lower_action(action))
            .collect::<Result<Vec<_>, _>>()
            .map(Block)
    }

    fn lower_action(&mut self, action: &Action<Param>) -> Result<Stmt, LowerError> {
        match action {
            Action::Loop {
                id,
                extent,
                guard,
                body,
            } => {
                let iter = loop_ident(*id);
                self.loops.insert(
                    *id,
                    LoopInfo {
                        iter: iter.clone(),
                        source: extent.source,
                        kind: extent.kind.clone(),
                    },
                );
                let mut body = self.lower_block(body)?;
                if guard.0 {
                    body = Block(vec![Stmt::If {
                        cond: Expr::Lt(
                            Box::new(self.reconstruct_extent_index(extent.source)?),
                            Box::new(param_dim(extent.source)),
                        ),
                        body,
                    }]);
                }
                self.loops.remove(id);

                Ok(Stmt::Loop {
                    iter,
                    bound: extent_expr(param_dim(extent.source), &extent.kind),
                    body,
                })
            }
            Action::Init { op, write } => Ok(Stmt::Set {
                dst: self.lower_write(write)?,
                value: init_value(*op)?,
            }),
            Action::Compute { op, write, reads } => {
                let dst = self.lower_write(write)?;
                let mut args = Vec::new();
                if self.initialized.contains(&write.buffer) {
                    args.push(place_expr(dst.clone()));
                }
                for read in reads {
                    args.push(self.lower_read(read)?);
                }
                Ok(Stmt::Set {
                    dst,
                    value: Expr::Op { op: *op, args },
                })
            }
        }
    }

    fn lower_write(&self, access: &Access<Param>) -> Result<Place, LowerError> {
        let buffer = param_place(access.buffer);
        let index = self.flat_index(param_expr(access.buffer), &access.index)?;
        Ok(index_place(field_place(buffer, Field::Data), index))
    }

    fn lower_read(&self, access: &Access<Param>) -> Result<Expr, LowerError> {
        let buffer = param_expr(access.buffer);
        let index = self.flat_index(buffer.clone(), &access.index)?;
        Ok(index_expr(field_expr(buffer, Field::Data), index))
    }

    fn flat_index(&self, buffer: Expr, index: &[Iter]) -> Result<Expr, LowerError> {
        if index.is_empty() {
            return Ok(Expr::Usize(0));
        }

        let dims = index
            .iter()
            .map(|iter| self.lower_iter(iter))
            .collect::<Result<Vec<_>, _>>()?;
        let mut sum = None;
        for (dim, expr) in dims.iter().cloned().enumerate() {
            let stride = ((dim + 1)..dims.len())
                .map(|layout_dim| {
                    index_expr(
                        field_expr(buffer.clone(), Field::Layout),
                        Expr::Usize(layout_dim),
                    )
                })
                .reduce(mul);
            let term = stride
                .map(|stride| mul(expr.clone(), stride))
                .unwrap_or(expr);
            sum = Some(sum.map(|sum| add(sum, term.clone())).unwrap_or(term));
        }
        Ok(sum.unwrap_or(Expr::Usize(0)))
    }

    fn lower_iter(&self, iter: &Iter) -> Result<Expr, LowerError> {
        match iter {
            Iter::Raw(loop_id) => self.loop_expr(*loop_id),
            Iter::Reconstructed { loops, factors } => {
                let iters = loops
                    .iter()
                    .copied()
                    .map(|loop_id| self.loop_expr(loop_id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(reconstruct_index(iters, factors))
            }
        }
    }

    fn loop_expr(&self, loop_id: LoopId) -> Result<Expr, LowerError> {
        self.loops
            .get(&loop_id)
            .map(|info| Expr::Ident(info.iter.clone()))
            .ok_or_else(|| LowerError::new(format!("loop {} is not in scope", loop_id.0)))
    }

    fn reconstruct_extent_index(&self, source: DimRef<Param>) -> Result<Expr, LowerError> {
        let mut infos = self
            .loops
            .values()
            .filter(|info| info.source == source)
            .cloned()
            .collect::<Vec<_>>();
        infos.sort_by_key(|info| extent_order(&info.kind));
        let iters = infos
            .iter()
            .map(|info| Expr::Ident(info.iter.clone()))
            .collect::<Vec<_>>();
        let factors = infos
            .iter()
            .find_map(|info| match &info.kind {
                ExtentKind::Base(factors) => Some(factors.clone()),
                _ => None,
            })
            .unwrap_or_else(|| {
                infos
                    .iter()
                    .filter_map(|info| match info.kind {
                        ExtentKind::Split { factor, .. } => Some(factor),
                        _ => None,
                    })
                    .collect()
            });
        Ok(reconstruct_index(iters, factors.as_slice()))
    }
}

#[derive(Clone)]
struct LoopInfo {
    iter: Ident,
    source: DimRef<Param>,
    kind: ExtentKind,
}

fn initialized_buffers(block: &KernelBlock<Param>) -> BTreeSet<Param> {
    let mut initialized = BTreeSet::new();
    collect_initialized(block, &mut initialized);
    initialized
}

fn collect_initialized(block: &KernelBlock<Param>, initialized: &mut BTreeSet<Param>) {
    for action in &block.0 {
        match action {
            Action::Loop { body, .. } => collect_initialized(body, initialized),
            Action::Init { write, .. } => {
                initialized.insert(write.buffer);
            }
            Action::Compute { .. } => {}
        }
    }
}

fn extent_expr(source: Expr, kind: &ExtentKind) -> Expr {
    match kind {
        ExtentKind::Semantic => source,
        ExtentKind::Base(factors) => {
            let factor = factors.iter().product::<usize>();
            if factor == 1 {
                source
            } else {
                div(
                    sub(add(source, Expr::Usize(factor)), Expr::Usize(1)),
                    Expr::Usize(factor),
                )
            }
        }
        ExtentKind::Split { factor, .. } => Expr::Usize(*factor),
    }
}

fn reconstruct_index(iters: Vec<Expr>, factors: &[usize]) -> Expr {
    if iters.is_empty() {
        return Expr::Usize(0);
    }
    if factors.is_empty() {
        return iters.into_iter().reduce(add).unwrap_or(Expr::Usize(0));
    }

    let mut terms = Vec::new();
    for (index, iter) in iters.into_iter().enumerate() {
        let weight = factors[index..].iter().product::<usize>();
        if weight == 1 {
            terms.push(iter);
        } else {
            terms.push(mul(iter, Expr::Usize(weight)));
        }
    }
    terms.into_iter().reduce(add).unwrap_or(Expr::Usize(0))
}

fn init_value(op: Op) -> Result<Expr, LowerError> {
    match op {
        Op::Add | Op::Or | Op::Xor => Ok(Expr::Scalar(0.0)),
        Op::Mul | Op::And => Ok(Expr::Scalar(1.0)),
        Op::Max => Ok(Expr::Scalar(f32::NEG_INFINITY)),
        Op::Min => Ok(Expr::Scalar(f32::INFINITY)),
        _ => Err(LowerError::new(format!(
            "op {:?} has no reduction identity",
            op
        ))),
    }
}

fn param_expr(param: Param) -> Expr {
    let (bucket, ind) = param_parts(param);
    index_expr(Expr::Ident(id(bucket)), Expr::Usize(ind))
}

fn param_place(param: Param) -> Place {
    let (bucket, ind) = param_parts(param);
    index_place(Place::Ident(id(bucket)), Expr::Usize(ind))
}

fn param_parts(param: Param) -> (&'static str, usize) {
    match param.arg {
        Arg::Readonly => ("readonlys", param.ind),
        Arg::Writeable => ("writeables", param.ind),
    }
}

fn param_dim(dim: DimRef<Param>) -> Expr {
    index_expr(
        field_expr(param_expr(dim.buffer), Field::Shape),
        Expr::Usize(dim.dim),
    )
}

fn place_expr(place: Place) -> Expr {
    match place {
        Place::Ident(ident) => Expr::Ident(ident),
        Place::Index { base, index } => index_expr(place_expr(*base), index),
        Place::Field { base, field } => field_expr(place_expr(*base), field),
    }
}

fn index_expr(base: Expr, index: Expr) -> Expr {
    Expr::Index {
        base: Box::new(base),
        index: Box::new(index),
    }
}

fn field_expr(base: Expr, field: Field) -> Expr {
    Expr::Field {
        base: Box::new(base),
        field,
    }
}

fn index_place(base: Place, index: Expr) -> Place {
    Place::Index {
        base: Box::new(base),
        index,
    }
}

fn field_place(base: Place, field: Field) -> Place {
    Place::Field {
        base: Box::new(base),
        field,
    }
}

fn add(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Add(Box::new(lhs), Box::new(rhs))
}

fn sub(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Sub(Box::new(lhs), Box::new(rhs))
}

fn mul(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Mul(Box::new(lhs), Box::new(rhs))
}

fn div(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Div(Box::new(lhs), Box::new(rhs))
}

fn extent_order(kind: &ExtentKind) -> (usize, usize) {
    match kind {
        ExtentKind::Semantic | ExtentKind::Base(_) => (0, 0),
        ExtentKind::Split { level, .. } => (1, *level),
    }
}

fn id(value: impl Into<String>) -> Ident {
    Ident(value.into())
}

fn kernel_ident(index: usize) -> Ident {
    id(format!("f{}", index))
}

fn input_ident(input: Input) -> Ident {
    id(format!("in{}", input.0))
}

fn intermediate_ident(intermediate: crate::ir::exec_plan::Intermediate) -> Ident {
    id(format!("s{}", intermediate.0))
}

fn output_ident(output: crate::ir::exec_plan::Output) -> Ident {
    id(format!("out{}", output.0))
}

fn loop_ident(loop_id: LoopId) -> Ident {
    id(format!("i{}", loop_id.0))
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

    fn from_exec_plan(error: crate::check::exec_plan::ValidationError) -> Self {
        Self::new(error.to_string())
    }

    fn from_module(error: crate::check::module::ValidationError) -> Self {
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
    use super::lower_exec_plan_to_module;
    use crate::front::parse_expr;
    use crate::ir::common::Op;
    use crate::ir::module::{Block, Cast, Expr, Field, Ident, Place, Signature, Stmt, Type};
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::kernel_program_to_exec_plan::lower_kernel_program_to_exec_plan;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;
    use crate::{component, front};

    fn lower_expr(src: &str) -> crate::ir::module::Module {
        let component = component::expr(parse_expr(src).unwrap());
        lower_component(&component)
    }

    fn lower_component(component: &crate::ir::component::Component) -> crate::ir::module::Module {
        let graph = lower_component_to_graph(component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        lower_exec_plan_to_module(&exec_plan).unwrap()
    }

    fn id(value: &str) -> Ident {
        Ident(value.to_string())
    }

    fn ident(value: &str) -> Expr {
        Expr::Ident(id(value))
    }

    fn index(base: Expr, index: Expr) -> Expr {
        Expr::Index {
            base: Box::new(base),
            index: Box::new(index),
        }
    }

    fn field(base: Expr, field: Field) -> Expr {
        Expr::Field {
            base: Box::new(base),
            field,
        }
    }

    #[test]
    fn lowers_public_abi_functions() {
        let module = lower_expr("ik*kj~ijk");

        assert_eq!(module.count.signature, Signature::Count);
        assert_eq!(module.ranks.signature, Signature::Ranks);
        assert_eq!(module.shapes.signature, Signature::Shapes);
        assert_eq!(module.exec.signature, Signature::Exec);
        assert_eq!(
            module.count.body,
            Block(vec![Stmt::Return(Some(Expr::Usize(1)))])
        );
        assert_eq!(
            module.ranks.body.0[0],
            Stmt::Set {
                dst: Place::Index {
                    base: Box::new(Place::Ident(id("ranks"))),
                    index: Expr::Usize(0)
                },
                value: Expr::Usize(3)
            }
        );
    }

    #[test]
    fn lowers_output_shapes_from_input_shape_fields() {
        let module = lower_expr("+ij~ji");

        assert_eq!(
            module.shapes.body.0[0],
            Stmt::Set {
                dst: Place::Index {
                    base: Box::new(Place::Index {
                        base: Box::new(Place::Ident(id("shapes"))),
                        index: Expr::Usize(0)
                    }),
                    index: Expr::Usize(0)
                },
                value: index(
                    field(index(ident("inputs"), Expr::Usize(0)), Field::Shape),
                    Expr::Usize(1)
                )
            }
        );
    }

    #[test]
    fn names_kernels_and_intermediates_by_convention() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let module = lower_component(&component);

        assert_eq!(module.kernels[0].ident, id("f0"));
        assert_eq!(module.kernels[1].ident, id("f1"));
        assert!(module
            .exec
            .body
            .0
            .iter()
            .any(|stmt| { matches!(stmt, Stmt::Alloc { dst, .. } if *dst == id("s0")) }));
        assert!(module.exec.body.0.iter().any(|stmt| {
            matches!(stmt, Stmt::Dispatch { kernel, reads, writes }
                if *kernel == id("f1")
                    && reads == &vec![id("s0")]
                    && writes == &vec![id("out0")])
        }));
    }

    #[test]
    fn lowers_exec_bindings_and_dispatch() {
        let module = lower_expr("ik*kj~ijk");

        assert_eq!(
            module.exec.body.0[0],
            Stmt::Let {
                ident: id("in0"),
                ty: Type::View,
                value: Expr::Cast {
                    kind: Cast::View,
                    value: Box::new(index(ident("inputs"), Expr::Usize(0)))
                }
            }
        );
        assert!(module.exec.body.0.iter().any(|stmt| {
            matches!(stmt, Stmt::Dispatch { kernel, reads, writes }
                if *kernel == id("f0")
                    && reads == &vec![id("in0"), id("in1")]
                    && writes == &vec![id("out0")])
        }));
    }

    #[test]
    fn lowers_alloc_shape_and_layout_expressions() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let module = lower_component(&component);
        let alloc = module
            .exec
            .body
            .0
            .iter()
            .find_map(|stmt| match stmt {
                Stmt::Alloc { dst, shape, layout } if *dst == id("s0") => Some((shape, layout)),
                _ => None,
            })
            .unwrap();

        assert_eq!(alloc.0.len(), 3);
        assert_eq!(
            alloc.0[0],
            index(
                field(index(ident("inputs"), Expr::Usize(0)), Field::Shape),
                Expr::Usize(0)
            )
        );
        assert_eq!(alloc.1.len(), 3);
    }

    #[test]
    fn lowers_split_loop_bound_to_ceil_div_arithmetic() {
        let module = lower_expr("ik*kj~ijk|i:8|ii'jk");
        let Stmt::Loop { bound, .. } = &module.kernels[0].body.0[0] else {
            unreachable!();
        };

        assert_eq!(
            *bound,
            Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Add(
                        Box::new(index(
                            field(index(ident("writeables"), Expr::Usize(0)), Field::Shape),
                            Expr::Usize(0)
                        )),
                        Box::new(Expr::Usize(8))
                    )),
                    Box::new(Expr::Usize(1))
                )),
                Box::new(Expr::Usize(8))
            )
        );
    }

    #[test]
    fn lowers_tail_guard_to_positive_if() {
        let module = lower_expr("ik*kj~ijk|i:8|ii'jk");
        let Stmt::Loop { body, .. } = &module.kernels[0].body.0[0] else {
            unreachable!();
        };
        let Stmt::Loop { body, .. } = &body.0[0] else {
            unreachable!();
        };
        let Stmt::If { cond, .. } = &body.0[0] else {
            unreachable!();
        };

        assert!(matches!(cond, Expr::Lt(_, _)));
    }

    #[test]
    fn lowers_flat_data_access_with_layout_fields() {
        let module = lower_expr("ik*kj~ijk");
        let compute = first_set_with_op(&module.kernels[0].body);
        let Stmt::Set { value, .. } = compute else {
            unreachable!();
        };
        let Expr::Op { args, .. } = value else {
            unreachable!();
        };

        assert!(contains_layout_field(&args[0]));
    }

    #[test]
    fn collapses_reduction_init_and_compute_to_set_statements() {
        let module = lower_expr("+ijk~ij");
        let mut sets = Vec::new();
        collect_sets(&module.kernels[0].body, &mut sets);

        assert!(sets.iter().any(|stmt| {
            matches!(
                stmt,
                Stmt::Set {
                    value: Expr::Scalar(0.0),
                    ..
                }
            )
        }));
        assert!(sets.iter().any(|stmt| {
            matches!(stmt, Stmt::Set {
                value: Expr::Op { op: Op::Add, args },
                ..
            } if args.len() == 2)
        }));
    }

    fn first_set_with_op(block: &Block) -> &Stmt {
        for stmt in &block.0 {
            match stmt {
                Stmt::Set {
                    value: Expr::Op { .. },
                    ..
                } => return stmt,
                Stmt::Loop { body, .. } | Stmt::If { body, .. } => return first_set_with_op(body),
                _ => {}
            }
        }
        unreachable!()
    }

    fn contains_layout_field(expr: &Expr) -> bool {
        match expr {
            Expr::Field {
                field: Field::Layout,
                ..
            } => true,
            Expr::Index { base, index } => {
                contains_layout_field(base) || contains_layout_field(index)
            }
            Expr::Field { base, .. } => contains_layout_field(base),
            Expr::Cast { value, .. } => contains_layout_field(value),
            Expr::Op { args, .. } => args.iter().any(contains_layout_field),
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Rem(lhs, rhs)
            | Expr::Lt(lhs, rhs) => contains_layout_field(lhs) || contains_layout_field(rhs),
            Expr::Usize(_) | Expr::Scalar(_) | Expr::Ident(_) => false,
        }
    }

    fn collect_sets<'a>(block: &'a Block, sets: &mut Vec<&'a Stmt>) {
        for stmt in &block.0 {
            match stmt {
                Stmt::Set { .. } => sets.push(stmt),
                Stmt::Loop { body, .. } | Stmt::If { body, .. } => collect_sets(body, sets),
                _ => {}
            }
        }
    }
}

use std::collections::BTreeMap;

use crate::ir::common::Op;
use crate::ir::module::{
    Block, Cast, Expr, Field, Fn, Ident, Module, Place, Signature, Stmt, Type,
};

type Env = BTreeMap<Ident, Type>;

pub fn render(module: &Module) -> String {
    let kernels = module
        .kernels
        .iter()
        .map(render_function)
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        r#"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {{ const float* data; const size_t* shape; const size_t rank; }} Tensor;
typedef struct {{ float* data; const size_t* shape; const size_t rank; }} TensorMut;
typedef struct {{ const float* data; const size_t* shape; const size_t* layout; }} View;
typedef struct {{ float* data; const size_t* shape; const size_t* layout; }} ViewMut;

static inline ViewMut alloc_view_mut(size_t ndims, const size_t* layout, const size_t* shape) {{
  size_t n = 1;
  for (size_t i = 0; i < ndims; ++i) n *= layout[i];
  float* data = (float*)malloc(n * sizeof(float));
  return (ViewMut){{ .data = data, .shape = shape, .layout = layout }};
}}

static inline View as_view(const Tensor* t) {{
  return (View){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline ViewMut as_view_mut(const TensorMut* t) {{
  return (ViewMut){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline View view_mut_as_view(const ViewMut* t) {{
  return (View){{ .data = t->data, .shape = t->shape, .layout = t->layout }};
}}

{count}

{ranks}

{shapes}

{kernels}

{exec}
"#,
        count = render_function(&module.count),
        ranks = render_function(&module.ranks),
        shapes = render_function(&module.shapes),
        kernels = kernels,
        exec = render_function(&module.exec),
    )
}

fn render_function(function: &Fn) -> String {
    let mut env = Env::new();
    format!(
        "{} {{\n{}\n}}",
        render_signature(function),
        indent(&render_block(&function.body, &mut env))
    )
}

fn render_signature(function: &Fn) -> String {
    match function.signature {
        Signature::Count => format!("size_t {}(void)", render_ident(&function.ident)),
        Signature::Ranks => format!("void {}(size_t* ranks)", render_ident(&function.ident)),
        Signature::Shapes => {
            format!(
                "void {}(const Tensor* inputs, size_t** shapes)",
                render_ident(&function.ident)
            )
        }
        Signature::Exec => {
            format!(
                "void {}(const Tensor* inputs, TensorMut* outputs)",
                render_ident(&function.ident)
            )
        }
        Signature::Kernel => {
            format!(
                "void {}(const View* readonlys, ViewMut* writeables)",
                render_ident(&function.ident)
            )
        }
    }
}

fn render_block(block: &Block, env: &mut Env) -> String {
    block
        .0
        .iter()
        .map(|stmt| render_stmt(stmt, env))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_stmt(stmt: &Stmt, env: &mut Env) -> String {
    match stmt {
        Stmt::Let { ident, ty, value } => {
            let rendered = format!(
                "{} {} = {};",
                render_type(ty),
                render_ident(ident),
                render_expr(value)
            );
            env.insert(ident.clone(), ty.clone());
            rendered
        }
        Stmt::Set { dst, value } => {
            format!("{} = {};", render_place(dst), render_expr(value))
        }
        Stmt::Alloc { dst, shape, layout } => {
            let rendered = render_alloc(dst, shape, layout);
            env.insert(dst.clone(), Type::ViewMut);
            rendered
        }
        Stmt::Free(ident) => {
            format!("free({}.data);", render_ident(ident))
        }
        Stmt::Dispatch {
            kernel,
            reads,
            writes,
        } => {
            format!(
                "{}({}, {});",
                render_ident(kernel),
                render_readonly_array(reads, env),
                render_writeable_array(writes, env)
            )
        }
        Stmt::Loop { iter, bound, body } => {
            let iter_ident = iter;
            let iter = render_ident(iter_ident);
            let mut child = env.clone();
            child.insert(iter_ident.clone(), Type::Usize);
            format!(
                "for (size_t {iter} = 0; {iter} < ({}); ++{iter}) {{\n{}\n}}",
                render_expr(bound),
                indent(&render_block(body, &mut child))
            )
        }
        Stmt::If { cond, body } => {
            let mut child = env.clone();
            format!(
                "if ({}) {{\n{}\n}}",
                render_expr(cond),
                indent(&render_block(body, &mut child))
            )
        }
        Stmt::Return(Some(expr)) => {
            format!("return {};", render_expr(expr))
        }
        Stmt::Return(None) => "return;".to_string(),
    }
}

fn render_alloc(dst: &Ident, shape: &[Expr], layout: &[Expr]) -> String {
    let ident = render_ident(dst);
    let layout_ident = format!("{ident}_layout");
    let shape_ident = format!("{ident}_shape");
    let layout_decl = render_array_decl(&layout_ident, layout);
    let shape_decl = render_array_decl(&shape_ident, shape);
    let layout_arg = if layout.is_empty() {
        "NULL".to_string()
    } else {
        layout_ident
    };
    let shape_arg = if shape.is_empty() {
        "NULL".to_string()
    } else {
        shape_ident
    };

    format!(
        "{layout_decl}\n{shape_decl}\nViewMut {ident} = alloc_view_mut({}, {layout_arg}, {shape_arg});",
        layout.len(),
    )
}

fn render_array_decl(ident: &str, values: &[Expr]) -> String {
    if values.is_empty() {
        format!("const size_t* {ident} = NULL;")
    } else {
        format!(
            "const size_t {ident}[] = {{ {} }};",
            values
                .iter()
                .map(render_expr)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn render_readonly_array(reads: &[Ident], env: &Env) -> String {
    format!(
        "(const View[]){{ {} }}",
        reads
            .iter()
            .map(|ident| render_readonly_arg(ident, env))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_writeable_array(writes: &[Ident], env: &Env) -> String {
    format!(
        "(ViewMut[]){{ {} }}",
        writes
            .iter()
            .map(|ident| render_writeable_arg(ident, env))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_readonly_arg(ident: &Ident, env: &Env) -> String {
    let rendered = render_ident(ident);
    match env.get(ident) {
        Some(Type::View) => rendered,
        Some(Type::ViewMut) => format!("view_mut_as_view(&{rendered})"),
        Some(ty) => panic!("dispatch read {} has type {:?}", ident.0, ty),
        None => panic!("dispatch read {} is unbound", ident.0),
    }
}

fn render_writeable_arg(ident: &Ident, env: &Env) -> String {
    match env.get(ident) {
        Some(Type::ViewMut) => render_ident(ident),
        Some(ty) => panic!("dispatch write {} has type {:?}", ident.0, ty),
        None => panic!("dispatch write {} is unbound", ident.0),
    }
}

fn render_expr(expr: &Expr) -> String {
    match expr {
        Expr::Usize(value) => value.to_string(),
        Expr::Scalar(value) => render_scalar(*value),
        Expr::Ident(ident) => render_ident(ident),
        Expr::Index { base, index } => {
            format!("{}[{}]", render_expr(base), render_expr(index))
        }
        Expr::Field { base, field } => {
            format!("{}.{}", render_expr(base), render_field(*field))
        }
        Expr::Cast { kind, value } => render_cast(*kind, value),
        Expr::Op { op, args } => render_op(*op, args),
        Expr::Add(lhs, rhs) => render_infix("+", lhs, rhs),
        Expr::Sub(lhs, rhs) => render_infix("-", lhs, rhs),
        Expr::Mul(lhs, rhs) => render_infix("*", lhs, rhs),
        Expr::Div(lhs, rhs) => render_infix("/", lhs, rhs),
        Expr::Rem(lhs, rhs) => render_infix("%", lhs, rhs),
        Expr::Lt(lhs, rhs) => render_infix("<", lhs, rhs),
        Expr::Eq(lhs, rhs) => render_infix("==", lhs, rhs),
        Expr::And(lhs, rhs) => render_infix("&&", lhs, rhs),
    }
}

fn render_place(place: &Place) -> String {
    match place {
        Place::Ident(ident) => render_ident(ident),
        Place::Index { base, index } => {
            format!("{}[{}]", render_place(base), render_expr(index))
        }
        Place::Field { base, field } => {
            format!("{}.{}", render_place(base), render_field(*field))
        }
    }
}

fn render_cast(kind: Cast, value: &Expr) -> String {
    match kind {
        Cast::View => format!("as_view(&{})", render_expr(value)),
        Cast::ViewMut => format!("as_view_mut(&{})", render_expr(value)),
        Cast::ReadOnly => format!("view_mut_as_view(&{})", render_expr(value)),
    }
}

fn render_op(op: Op, args: &[Expr]) -> String {
    match op {
        Op::Add => render_joined("+", args),
        Op::Mul => render_joined("*", args),
        Op::Div => {
            if args.len() == 1 {
                format!("(1.0f / {})", render_expr(&args[0]))
            } else {
                render_joined("/", args)
            }
        }
        Op::Sub => {
            if args.len() == 1 {
                format!("(-{})", render_expr(&args[0]))
            } else {
                render_joined("-", args)
            }
        }
        Op::Max => render_call_fold("fmaxf", args),
        Op::Min => render_call_fold("fminf", args),
        Op::Pow => {
            if args.len() == 1 {
                render_unary_call("expf", args)
            } else {
                render_call_fold("powf", args)
            }
        }
        Op::Log => render_unary_call("logf", args),
        Op::Gt => render_joined(">", args),
        Op::Ge => render_joined(">=", args),
        Op::Lt => render_joined("<", args),
        Op::Le => render_joined("<=", args),
        Op::Eq => render_joined("==", args),
        Op::Ne => render_joined("!=", args),
        Op::And => render_joined("&&", args),
        Op::Or => render_joined("||", args),
        Op::Xor => render_xor(args),
        Op::Not => format!("(!{})", render_expr(&args[0])),
    }
}

fn render_joined(op: &str, args: &[Expr]) -> String {
    format!(
        "({})",
        args.iter()
            .map(render_expr)
            .collect::<Vec<_>>()
            .join(&format!(" {op} "))
    )
}

fn render_unary_call(function: &str, args: &[Expr]) -> String {
    format!("{function}({})", render_expr(&args[0]))
}

fn render_call_fold(function: &str, args: &[Expr]) -> String {
    let mut iter = args.iter().map(render_expr);
    let Some(first) = iter.next() else {
        return "0.0f".to_string();
    };
    iter.fold(first, |acc, arg| format!("{function}({acc}, {arg})"))
}

fn render_xor(args: &[Expr]) -> String {
    let mut iter = args.iter().map(render_expr);
    let Some(first) = iter.next() else {
        return "0".to_string();
    };
    iter.fold(first, |acc, arg| {
        format!("((({acc}) != 0.0f) != (({arg}) != 0.0f))")
    })
}

fn render_infix(op: &str, lhs: &Expr, rhs: &Expr) -> String {
    format!("({} {op} {})", render_expr(lhs), render_expr(rhs))
}

fn render_field(field: Field) -> &'static str {
    match field {
        Field::Data => "data",
        Field::Shape => "shape",
        Field::Layout => "layout",
        Field::Rank => "rank",
    }
}

fn render_type(ty: &Type) -> String {
    match ty {
        Type::Usize => "size_t".to_string(),
        Type::Scalar => "float".to_string(),
        Type::Tensor => "Tensor".to_string(),
        Type::TensorMut => "TensorMut".to_string(),
        Type::View => "const View".to_string(),
        Type::ViewMut => "ViewMut".to_string(),
        Type::Array(inner) => render_type(inner),
    }
}

fn render_scalar(value: f32) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value == f32::INFINITY {
        "INFINITY".to_string()
    } else if value == f32::NEG_INFINITY {
        "(-INFINITY)".to_string()
    } else {
        let value = format!("{:.8}", value);
        let value = value.trim_end_matches('0').trim_end_matches('.');
        if value.contains('.') {
            format!("{value}f")
        } else {
            format!("{value}.0f")
        }
    }
}

fn render_ident(ident: &Ident) -> String {
    ident.0.clone()
}

fn indent(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("  {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::front::parse_expr;
    use crate::ir::module::{Block, Cast, Expr, Fn, Ident, Module, Signature, Stmt, Type};
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::exec_plan_to_module::lower_exec_plan_to_module;
    use crate::lower::kernel_program_to_exec_plan::lower_kernel_program_to_exec_plan;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;
    use crate::{component, front};

    fn id(value: &str) -> Ident {
        Ident(value.to_string())
    }

    fn function(name: &str, signature: Signature, body: Block) -> Fn {
        Fn {
            ident: id(name),
            signature,
            body,
        }
    }

    fn render_expr(src: &str) -> String {
        let component = component::expr(parse_expr(src).unwrap());
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        let module = lower_exec_plan_to_module(&exec_plan).unwrap();
        render(&module)
    }

    #[test]
    fn renders_public_abi() {
        let c = render_expr("ik*kj~ijk");

        assert!(c.contains("size_t count(void)"));
        assert!(c.contains("void ranks(size_t* ranks)"));
        assert!(c.contains("void shapes(const Tensor* inputs, size_t** shapes)"));
        assert!(c.contains("void exec(const Tensor* inputs, TensorMut* outputs)"));
        assert!(c.contains("void f0(const View* readonlys, ViewMut* writeables)"));
    }

    #[test]
    fn renders_shapes_and_ranks() {
        let c = render_expr("+ij~ji");

        assert!(c.contains("ranks[0] = 2;"));
        assert!(c.contains("shapes[0][0] = inputs[0].shape[1];"));
        assert!(c.contains("shapes[0][1] = inputs[0].shape[0];"));
    }

    #[test]
    fn renders_exec_bindings_and_dispatch() {
        let c = render_expr("ik*kj~ijk");

        assert!(c.contains("const View in0 = as_view(&inputs[0]);"));
        assert!(c.contains("ViewMut out0 = as_view_mut(&outputs[0]);"));
        assert!(c.contains("f0((const View[]){ in0, in1 }, (ViewMut[]){ out0 });"));
    }

    #[test]
    fn renders_intermediate_alloc_and_free() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        let module = lower_exec_plan_to_module(&exec_plan).unwrap();
        let c = render(&module);

        assert!(c.contains("ViewMut s0 = alloc_view_mut("));
        assert!(c.contains("f1((const View[]){ view_mut_as_view(&s0) }, (ViewMut[]){ out0 });"));
        assert!(c.contains("free(s0.data);"));
    }

    #[test]
    fn renders_split_bounds_and_positive_guards() {
        let c = render_expr("ik*kj~ijk|i:8|ii'jk");

        assert!(c.contains("for (size_t i0 = 0; i0 <"));
        assert!(c.contains("writeables[0].shape[0] + 8"));
        assert!(c.contains("- 1) / 8"));
        assert!(c.contains("if ("));
        assert!(c.contains("i0 * 8"));
        assert!(c.contains("i1"));
        assert!(c.contains("< writeables[0].shape[0]"));
    }

    #[test]
    fn renders_data_access_and_ops() {
        let c = render_expr("+ijk~ij");

        assert!(c.contains(".data["));
        assert!(c.contains(".layout["));
        assert!(c.contains("= 0.0f;"));
        assert!(c.contains(" + "));
    }

    #[test]
    fn renders_unary_pow_as_exp() {
        let c = render_expr("^ij~ij");

        assert!(c.contains("expf("));
    }

    #[test]
    fn renders_zero_check_conjunction() {
        let c = render_expr("+ijkl~ij||ijkl!");

        assert!(c.contains(" == 0"));
        assert!(c.contains(" && "));
    }

    #[test]
    fn renders_dispatch_reads_from_bound_types() {
        let module = Module {
            count: function(
                "count",
                Signature::Count,
                Block(vec![Stmt::Return(Some(Expr::Usize(1)))]),
            ),
            ranks: function("ranks", Signature::Ranks, Block(vec![Stmt::Return(None)])),
            shapes: function("shapes", Signature::Shapes, Block(vec![Stmt::Return(None)])),
            exec: function(
                "exec",
                Signature::Exec,
                Block(vec![
                    Stmt::Let {
                        ident: id("source"),
                        ty: Type::ViewMut,
                        value: Expr::Cast {
                            kind: Cast::ViewMut,
                            value: Box::new(Expr::Index {
                                base: Box::new(Expr::Ident(id("outputs"))),
                                index: Box::new(Expr::Usize(0)),
                            }),
                        },
                    },
                    Stmt::Dispatch {
                        kernel: id("f0"),
                        reads: vec![id("source")],
                        writes: vec![id("source")],
                    },
                    Stmt::Return(None),
                ]),
            ),
            kernels: vec![function("f0", Signature::Kernel, Block(vec![]))],
        };
        let c = render(&module);

        assert!(
            c.contains("f0((const View[]){ view_mut_as_view(&source) }, (ViewMut[]){ source });")
        );
    }
}

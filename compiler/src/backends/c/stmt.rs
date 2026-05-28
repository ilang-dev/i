use std::collections::BTreeMap;

use crate::ir::module::{Block, Expr, Fn, Ident, Signature, Stmt, Type};

use super::expr::{render_expr, render_place};
use super::fmt::{indent, initializer};
use super::render_ident;

type Env = BTreeMap<Ident, Type>;

pub(super) fn render_function(function: &Fn) -> String {
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
                "{}(\n{},\n{}\n);",
                render_ident(kernel),
                indent(&render_readonly_array(reads, env)),
                indent(&render_writeable_array(writes, env))
            )
        }
        Stmt::Loop { iter, bound, body } => render_loop(iter, bound, body, env),
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

fn render_loop(iter: &Ident, bound: &Expr, body: &Block, env: &Env) -> String {
    let iter_name = render_ident(iter);
    let mut child = env.clone();
    child.insert(iter.clone(), Type::Usize);
    format!(
        "for (size_t {iter_name} = 0; {iter_name} < {}; ++{iter_name}) {{\n{}\n}}",
        render_expr(bound),
        indent(&render_block(body, &mut child))
    )
}

fn render_alloc(dst: &Ident, shape: &[Expr], layout: &[Expr]) -> String {
    let ident = render_ident(dst);
    let layout_ident = format!("{ident}_layout");
    let shape_ident = format!("{ident}_shape");
    let layout_arg = array_arg(&layout_ident, layout);
    let shape_arg = array_arg(&shape_ident, shape);

    format!(
        "{}\n{}\nViewMut {ident} = alloc_view_mut({}, {layout_arg}, {shape_arg});",
        render_array_decl(&layout_ident, layout),
        render_array_decl(&shape_ident, shape),
        layout.len(),
    )
}

fn render_array_decl(ident: &str, values: &[Expr]) -> String {
    if values.is_empty() {
        format!("const size_t* {ident} = NULL;")
    } else {
        format!(
            "const size_t {ident}[] = {};",
            initializer("", values.iter().map(render_expr))
        )
    }
}

fn array_arg(ident: &str, values: &[Expr]) -> String {
    if values.is_empty() {
        "NULL".to_string()
    } else {
        ident.to_string()
    }
}

fn render_readonly_array(reads: &[Ident], env: &Env) -> String {
    initializer(
        "(const View[])",
        reads.iter().map(|ident| render_readonly_arg(ident, env)),
    )
}

fn render_writeable_array(writes: &[Ident], env: &Env) -> String {
    initializer(
        "(ViewMut[])",
        writes.iter().map(|ident| render_writeable_arg(ident, env)),
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

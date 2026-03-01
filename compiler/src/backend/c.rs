use std::fs;
use std::io::Error;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::{Backend, Build, Render};
use crate::block::{Block, Expr, FunctionSignature, Program, Statement, Type};

fn unique_string() -> String {
    format!(
        "{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

pub struct CBackend;

impl Backend for CBackend {}

impl Build for CBackend {
    fn build(source: &str) -> Result<PathBuf, Error> {
        let path_base = "/tmp/ilang";
        let source_path = format!("{path_base}.c");
        let dylib_path = format!("{path_base}_{}.so", unique_string());
        fs::write(&source_path, source)?;
        let build = Command::new("cc")
            .args([
                "-O3",
                "-shared",
                "-fPIC",
                &source_path,
                "-o",
                &dylib_path,
                "-lm",
            ])
            .status();
        if let Err(e) = build {
            return Err(e);
        }
        let exit = build.unwrap();
        if !exit.success() {
            return Err(Error::last_os_error());
        }
        Ok(PathBuf::from(dylib_path))
    }
}

impl Render for CBackend {
    fn render(program: &Program) -> String {
        format!(
            r#"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct {{
    const float* data;
    const size_t* shape;
    const size_t rank;
}} Tensor;

typedef struct {{
    float* data;
    const size_t* shape;
    const size_t rank;
}} TensorMut;

typedef struct {{
    const float* data;
    const size_t* shape;
    const size_t* layout;
}} View;

typedef struct {{
    float* data;
    const size_t* shape;
    const size_t* layout;
}} ViewMut;

static inline ViewMut alloc_view_mut(float v, size_t ndims, const size_t* layout, const size_t* shape) {{
    size_t n = 1;
    for (size_t i = 0; i < ndims; ++i) n *= layout[i];
    float* data = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) data[i] = v;
    return (ViewMut){{ .data = data, .shape = shape, .layout = layout }};
}}

static inline const View as_view(const Tensor* t) {{
  return (const View){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline const ViewMut as_view_mut(const TensorMut* t) {{
  return (const ViewMut){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline const View view_mut_as_view(const ViewMut* t) {{
  return (const View){{ .data=t->data, .shape=t->shape, .layout=t->layout }};
}}

{count}
{ranks}
{shapes}
{library}
{exec}
"#,
            count = Self::render_statement(&program.count),
            ranks = Self::render_statement(&program.ranks),
            shapes = Self::render_statement(&program.shapes),
            library = Self::render_block(&program.library),
            exec = Self::render_statement(&program.exec)
        )
    }
}

impl CBackend {
    fn render_block(block: &Block) -> String {
        block
            .statements
            .iter()
            .map(|statement| Self::render_statement(&statement))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_type(t: &Type) -> String {
        match t {
            Type::Int(_) => "size_t".to_string(),
            Type::Scalar(_) => "float".to_string(),
            Type::View(mutable) => format!("{}View", if *mutable { "" } else { "const " }),
            Type::ViewMut(_) => "ViewMut".to_string(),
            Type::Array(type_) => Self::render_type(type_),
        }
    }

    fn render_op(expr: &Expr) -> String {
        let Expr::Op { op, inputs } = expr else {
            panic!("Expected Op")
        };
        match op {
            '!' => {
                assert!(inputs.len() == 1);
                let x = Self::render_expr(&inputs[0]);
                format!("(({x}) > 0.0f ? ({x}) : 0.0f)")
            }
            '>' => {
                assert!(inputs.len() == 2);
                let a = Self::render_expr(&inputs[0]);
                let b = Self::render_expr(&inputs[1]);
                format!("(({a}) > ({b}) ? ({a}) : ({b}))")
            }
            '<' => {
                assert!(inputs.len() == 2);
                let a = Self::render_expr(&inputs[0]);
                let b = Self::render_expr(&inputs[1]);
                format!("(({a}) < ({b}) ? ({a}) : ({b}))")
            }
            '^' => {
                assert!(inputs.len() == 1);
                format!("expf({})", Self::render_expr(&inputs[0]))
            }
            '$' => {
                assert!(inputs.len() == 1);
                format!("logf({})", Self::render_expr(&inputs[0]))
            }
            '@' => {
                assert!(inputs.len() == 1);
                format!("sqrtf({})", Self::render_expr(&inputs[0]))
            }
            '#' => {
                assert!(inputs.len() == 1);
                format!("fabsf({})", Self::render_expr(&inputs[0]))
            }
            c => {
                if inputs.len() == 1 {
                    if *c == '-' {
                        return format!("-( {})", Self::render_expr(&inputs[0]));
                    }
                    if *c == '/' {
                        return format!("(1.0f / {})", Self::render_expr(&inputs[0]));
                    }
                }
                let joined = inputs
                    .iter()
                    .map(|i| Self::render_expr(i))
                    .collect::<Vec<_>>()
                    .join(&format!(" {c} "));
                format!("({joined})")
            }
        }
    }

    fn render_float_literal(x: f32) -> String {
        if x.is_nan() {
            "NAN".to_string()
        } else if x == f32::INFINITY {
            "INFINITY".to_string()
        } else if x == f32::NEG_INFINITY {
            "(-INFINITY)".to_string()
        } else {
            let s = format!("{:.8}", x);
            let trimmed = if s.contains('.') {
                s.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                s
            };
            if trimmed.contains('.') {
                format!("{trimmed}f")
            } else {
                format!("{trimmed}.0f")
            }
        }
    }

    fn render_expr(expr: &Expr) -> String {
        match expr {
            Expr::Ident(s) => s.to_string(),
            Expr::Int(x) => format!("{}", x),
            Expr::Scalar(x) => Self::render_float_literal(*x),
            Expr::Array(type_, exprs) => format!(
                "{{{}}}",
                exprs
                    .iter()
                    .map(|expr| Self::render_expr(expr))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Expr::Op { .. } => Self::render_op(expr),
            Expr::Indexed { expr, index } => {
                format!("{}[{}]", Self::render_expr(expr), Self::render_expr(index))
            }
            Expr::DataOf(expr) => format!("{}.data", Self::render_expr(expr)),
            Expr::ShapeOf(expr) => format!("{}.shape", Self::render_expr(expr)),
            Expr::LayoutOf(expr) => format!("{}.layout", Self::render_expr(expr)),
            Expr::View(expr) => format!("as_view(&{})", Self::render_expr(expr)),
            Expr::ViewMut(expr) => format!("as_view_mut(&{})", Self::render_expr(expr)),
            Expr::ReadOnly(expr) => format!("view_mut_as_view(&{})", Self::render_expr(expr)),
        }
    }

    fn render_function_signature(signature: &FunctionSignature) -> String {
        match signature {
            FunctionSignature::Count => format!("size_t count()"),
            FunctionSignature::Ranks => format!("void ranks(size_t* ranks)"),
            FunctionSignature::Shapes => {
                format!("void shapes(const Tensor* inputs, size_t** shapes)")
            }
            FunctionSignature::Exec => {
                format!("void exec(const Tensor* inputs, TensorMut* outputs)")
            }
            FunctionSignature::Kernel(ident) => format!(
                "void {}(const View* inputs, ViewMut* outputs)",
                Self::render_expr(ident)
            ),
        }
    }

    fn render_ref_list(exprs: &Vec<Expr>, mutable: bool) -> String {
        format!(
            "({}View{} []){{{}}}",
            if mutable { "" } else { "const " },
            if mutable { "Mut" } else { "" },
            exprs
                .iter()
                .map(|expr| format!("{}", Self::render_expr(expr)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn render_statement(statement: &Statement) -> String {
        match statement {
            Statement::Assignment { left, right } => {
                format!(
                    "{} = {};",
                    Self::render_expr(left),
                    Self::render_expr(right)
                )
            }
            Statement::IfZero { indices, body } => {
                let condition = indices
                    .iter()
                    .map(|index| format!("({}) == (0)", Self::render_expr(index)))
                    .collect::<Vec<_>>()
                    .join(" && ");
                format!("if ({condition}) {{\n{}\n}}", Self::render_block(body))
            }
            Statement::Alloc {
                index,
                initial_value,
                layout,
                shape,
            } => {
                let layout_decl = match layout.is_empty() {
                    true => format!("const size_t* layout{index} = NULL;"),
                    false => {
                        let layout_init = layout
                            .iter()
                            .map(|dim| Self::render_expr(dim))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("const size_t layout{index}[] = {{ {layout_init} }};")
                    }
                };
                let shape_decl = match shape.is_empty() {
                    true => format!("const size_t* shape{index} = NULL;"),
                    false => {
                        let shape_init = shape
                            .iter()
                            .map(|dim| Self::render_expr(dim))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("const size_t shape{index}[] = {{ {shape_init} }};")
                    }
                };
                format!(
                    "
                        {layout_decl}
                        {shape_decl}
                        ViewMut s{index} = alloc_view_mut({}, {}, layout{index}, shape{index});
                    ",
                    Self::render_expr(initial_value),
                    layout.len(),
                )
            }
            Statement::Declaration {
                ident,
                value,
                type_,
            } => {
                let ty = Self::render_type(type_);
                format!(
                    "{ty} {}{} = {};",
                    Self::render_expr(ident),
                    if let Type::Array(_) = type_ { "[]" } else { "" },
                    Self::render_expr(value)
                )
            }
            Statement::Skip { index, bound } => format!(
                "if (({}) >= ({})) {{ continue; }}",
                Self::render_expr(index),
                Self::render_expr(bound),
            ),
            Statement::Loop {
                index,
                bound,
                body,
                parallel: _,
            } => {
                let index: String = Self::render_expr(index);
                format!(
                    "for (size_t {index} = 0; {index} < ({}); ++{index}) {{\n{}\n}}",
                    Self::render_expr(bound),
                    Self::render_block(body)
                )
            }
            Statement::Function { signature, body } => format!(
                "{}{{{}}}",
                Self::render_function_signature(signature),
                Self::render_block(body)
            ),
            Statement::Return { value } => {
                format!("return {};", Self::render_expr(value))
            }
            Statement::Call {
                ident,
                in_args,
                out_args,
            } => {
                format!(
                    "{}({}, {});",
                    Self::render_expr(ident),
                    Self::render_ref_list(&in_args, false),
                    Self::render_ref_list(&out_args, true),
                )
            }
        }
    }
}

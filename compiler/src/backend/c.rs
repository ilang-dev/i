use std::collections::HashMap;
use std::fs;
use std::io::Error;
use std::path::PathBuf;
use std::process::Command;

use crate::backend::{Backend, Build, Render};
use crate::block::{Arg, Block, Expr, Program, Statement, Type};

use std::time::{SystemTime, UNIX_EPOCH};

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
    size_t ndim;
}} Tensor;

typedef struct {{
    float* data;
    const size_t* shape;
    size_t ndim;
}} TensorMut;

static inline float* ilang_alloc_f32(size_t n, float v) {{
    float* p = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) p[i] = v;
    return p;
}}

{rank}
{shape}
{library}
{exec}
"#,
            rank = Self::render_rank(&program.rank),
            shape = Self::render_shape(&program.shape),
            library = Self::render_block(&program.library),
            exec = Self::render_exec(&program.exec)
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
            Type::Array(m) | Type::ArrayRef(m) => {
                if *m {
                    "float*".to_string()
                } else {
                    "const float*".to_string()
                }
            }
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
            Expr::Alloc {
                initial_value,
                shape,
            } => {
                format!(
                    "ilang_alloc_f32((size_t)({}), ({}))",
                    shape.join(" * "),
                    Self::render_expr(initial_value)
                )
            }
            Expr::Ident(s) => s.to_string(),
            Expr::Ref(s, _mutable) => s.to_string(),
            Expr::Int(x) => format!("{}", x),
            Expr::Scalar(x) => Self::render_float_literal(*x),
            Expr::Op { .. } => Self::render_op(expr),
            Expr::Indexed { ident, index } => {
                format!("{ident}[{}]", Self::render_expr(index))
            }
        }
    }

    fn render_rank(statement: &Statement) -> String {
        if let Statement::Function { body, .. } = statement {
            format!("size_t rank() {{\n{}\n}}\n", Self::render_block(body))
        } else {
            panic!("Non-function for rank")
        }
    }

    fn render_shape(statement: &Statement) -> String {
        if let Statement::Function { args, body, .. } = statement {
            let mut inputs_arrays = args
                .iter()
                .filter(|a| matches!(a.type_, Type::ArrayRef(_)))
                .collect::<Vec<_>>();
            let _ = inputs_arrays.pop();

            let n_input_arrays = inputs_arrays.len();
            let input_shape_vecs_string = (0..n_input_arrays)
                .map(|ind| format!("const size_t* d{ind} = inputs[{ind}].shape;"))
                .collect::<Vec<_>>()
                .join("\n");

            let output_shape_vec_string = "size_t* shape_vec = (size_t*)shape;";

            format!(
                "void shape(const Tensor* inputs, size_t n_inputs, size_t rank_val, size_t* shape) {{\n{input_shape_vecs_string}\n{output_shape_vec_string}\n{}\n}}\n",
                Self::render_block(body)
            )
        } else {
            panic!("Non-function for shape")
        }
    }

    fn render_exec(statement: &Statement) -> String {
        if let Statement::Function { args, body, .. } = statement {
            let mut inputs_arrays = args
                .iter()
                .filter(|a| matches!(a.type_, Type::ArrayRef(_)))
                .collect::<Vec<_>>();
            let _ = inputs_arrays.pop();

            let n_input_arrays = inputs_arrays.len();
            let input_shape_vecs_string = (0..n_input_arrays)
                .map(|ind| format!("const size_t* d{ind} = inputs[{ind}].shape;"))
                .collect::<Vec<_>>()
                .join("\n");

            let input_arrays_string = (0..n_input_arrays)
                .map(|ind| format!("const float* in{ind} = inputs[{ind}].data;"))
                .collect::<Vec<_>>()
                .join("\n");

            let mut bound_variable_string = String::new();
            let mut array_arg_ind = 0usize;
            let mut array_dim_ind = 0usize;
            let mut n_inputs_arrays_seen = 0usize;

            for arg in args {
                match arg.type_ {
                    Type::ArrayRef(_) => {
                        array_arg_ind += 1;
                        array_dim_ind = 0;
                        n_inputs_arrays_seen = array_arg_ind - 1;
                    }
                    Type::Int(_) => {
                        array_dim_ind += 1;
                        let shape_vec_string = if array_arg_ind > n_input_arrays {
                            "dout".to_string()
                        } else {
                            format!("d{}", n_inputs_arrays_seen)
                        };
                        let Expr::Ident(ref bound_ident) = arg.ident else {
                            panic!()
                        };
                        bound_variable_string.push_str(&format!(
                            "size_t {bound_ident} = {shape_vec_string}[{}];\n",
                            array_dim_ind - 1
                        ));
                    }
                    _ => {}
                }
            }

            let output_shape_vec_string = "const size_t* dout = output->shape;";
            let output_array_string = "float* out = output->data;";

            format!(
                "void f(const Tensor* inputs, size_t n_inputs, TensorMut* output) {{\n{input_shape_vecs_string}\n{output_shape_vec_string}\n{input_arrays_string}\n{output_array_string}\n{bound_variable_string}\n{}\n}}\n",
                Self::render_block(body)
            )
        } else {
            panic!("Non-function for exec")
        }
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
            Statement::Declaration {
                ident,
                value,
                type_,
            } => {
                let ty = Self::render_type(type_);
                format!("{ty} {ident} = {};", Self::render_expr(value))
            }
            Statement::Skip { index, bound } => {
                format!("if (({}) >= ({})) {{ continue; }}", index, bound)
            }
            Statement::Loop {
                index, bound, body, ..
            } => {
                format!(
                    "for (size_t {index} = 0; {index} < ({}); ++{index}) {{\n{}\n}}",
                    Self::render_expr(bound),
                    Self::render_block(body)
                )
            }
            Statement::Function { ident, args, body } => {
                let sig = args
                    .iter()
                    .map(|Arg { type_, ident }| {
                        format!("{} {}", Self::render_type(type_), Self::render_expr(ident))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("void {ident}({sig}) {{\n{}\n}}\n", Self::render_block(body))
            }
            Statement::Return { value } => {
                format!("return {};", Self::render_expr(value))
            }
            Statement::Call { ident, args } => {
                let a = args
                    .iter()
                    .map(|Arg { ident, .. }| Self::render_expr(ident))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{ident}({});", a)
            }
        }
    }
}

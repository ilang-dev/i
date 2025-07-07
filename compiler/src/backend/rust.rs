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

pub struct RustBackend;

impl Backend for RustBackend {}

impl Build for RustBackend {
    fn build(source: &str) -> Result<PathBuf, Error> {
        let path_base = "/tmp/ilang";
        let source_path = format!("{path_base}.rs");
        let dylib_path = format!("{path_base}_{}.so", unique_string());
        fs::write(&source_path, source)?;
        let build = Command::new("rustc")
            .args([
                "--crate-type=dylib",
                "-C",
                "opt-level=3",
                &source_path,
                "-o",
                &dylib_path,
                "-A",
                "warnings",
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

impl Render for RustBackend {
    fn render(program: &Program) -> String {
        format!(
            r#"
#[repr(C)]
struct Tensor<'a> {{
    data: *const f32,
    shape: *const usize,
    ndim: usize,
    _marker: std::marker::PhantomData<&'a [f32]>,
}}

#[repr(C)]
struct TensorMut<'a> {{
    data: *mut f32,
    shape: *const usize,
    ndim: usize,
    _marker: std::marker::PhantomData<&'a mut [f32]>,
}}

{}

{}

{}

{}
"#,
            Self::render_rank(&program.rank),
            Self::render_shape(&program.shape),
            Self::render_block(&program.library),
            Self::render_exec(&program.exec)
        )
    }
}

impl RustBackend {
    fn render_block(block: &Block) -> String {
        block
            .statements
            .iter()
            .map(|statement| Self::render_statement(&statement))
            .collect::<Vec<_>>()
            .join("\n")
    }
    fn render_type(type_: &Type) -> String {
        match type_ {
            Type::Int(_) => "usize".to_string(),
            Type::Array(mutable) | Type::ArrayRef(mutable) => {
                format!("&{}[f32]", if *mutable { "mut " } else { "" })
            }
        }
    }
    fn render_op(expr: &Expr) -> String {
        let Expr::Op { op, inputs } = expr else {
            panic!("Expected `Op` variant of `Expr`")
        };
        match op {
            '>' => match inputs.len() {
                1 => format!(
                    "if {} > 0. {{ {} }} else {{ 0. }}",
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[0]),
                ),
                2 => format!(
                    "if {} > {} {{ {} }} else {{ {} }}",
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[1]),
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[1]),
                ),
                _ => panic!("Expected 1 or 2 inputs to op [>]."),
            },
            '^' => {
                assert!(inputs.len() == 1, "Expected 1 input to op [^].");
                format!("{}.exp()", Self::render_expr(&inputs[0]))
            }
            '$' => {
                assert!(inputs.len() == 1, "Expected 1 input to op [$].");
                format!("{}.ln()", Self::render_expr(&inputs[0]))
            }
            '@' => {
                assert!(inputs.len() == 1, "Expected 1 input to op [$].");
                format!("{}.sqrt()", Self::render_expr(&inputs[0]))
            }
            '#' => {
                assert!(inputs.len() == 1, "Expected 1 input to op [$].");
                format!("{}.abs()", Self::render_expr(&inputs[0]))
            }
            c => {
                if inputs.len() == 1 {
                    if *c == '-' {
                        return format!("-({})", Self::render_expr(&inputs[0]));
                    }
                    if *c == '/' {
                        return format!("1. / {}", Self::render_expr(&inputs[0]));
                    }
                }
                format!(
                    "({})",
                    inputs
                        .iter()
                        .map(|input| Self::render_expr(&input))
                        .collect::<Vec<_>>()
                        .join(&format!(" {op} "))
                )
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
                    "&mut vec![{}; {}][..]",
                    Self::render_expr(&initial_value),
                    format!("{}", shape.join(" * ")),
                )
            }
            Expr::Ident(s) => s.to_string(),
            Expr::Ref(s, _mutable) => format!("{s}"),
            Expr::Int(x) => format!("{:.1}", x), // using `.to_string()` won't produce decimal
            Expr::Scalar(x) => match x {
                x if x.is_nan() => "f32::NAN".to_string(),
                x if *x == f32::INFINITY => "f32::INFINITY".to_string(),
                x if *x == f32::NEG_INFINITY => "f32::NEG_INFINITY".to_string(),
                x => {
                    let s = format!("{:.8}", x);
                    if s.contains('.') {
                        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
                        format!(
                            "{}{}",
                            trimmed,
                            if trimmed.contains('.') { "" } else { ".0" }
                        )
                    } else {
                        format!("{}.", s)
                    }
                }
            },
            Expr::Op { .. } => Self::render_op(&expr),
            Expr::Indexed { ident, index } => format!("{ident}[{}]", Self::render_expr(&index),),
        }
    }

    fn render_rank(statement: &Statement) -> String {
        if let Statement::Function { body, .. } = &statement {
            format!(
                r#"
#[no_mangle]
extern "C"
fn rank() -> usize {{
    {function_body}
}}
"#,
                function_body = Self::render_block(&body),
            )
        } else {
            panic!("Found non-`Function` `Statement` for executive function.")
        }
    }

    fn render_shape(statement: &Statement) -> String {
        if let Statement::Function { ident, args, body } = &statement {
            let mut inputs_arrays = args
                .iter()
                .filter(|arg| matches!(arg.type_, Type::ArrayRef(_)))
                .collect::<Vec<_>>();

            let inputs_dims = args
                .iter()
                .filter(|arg| matches!(arg.type_, Type::Int(_)))
                .collect::<Vec<_>>();

            let _ = inputs_arrays.pop(); // TODO: What if there are no inputs?

            let n_input_arrays = inputs_arrays.len();
            let input_shape_vecs_string = (0..n_input_arrays)
                .map(|ind| format!(
                    "let d{ind} = std::slice::from_raw_parts(inputs[{ind}].shape, inputs[{ind}].ndim);"
                ))
                .collect::<Vec<_>>()
                .join("\n");

            let output_shape_vec_string =
                "let shape = std::slice::from_raw_parts_mut(shape, rank);";

            format!(
                r#"
#[no_mangle]
unsafe extern "C"
fn shape(inputs: *const Tensor, n_inputs: usize, rank: usize, shape: *mut usize) {{
    let inputs = std::slice::from_raw_parts(inputs, n_inputs);

    {input_shape_vecs_string}
    {output_shape_vec_string}

    {function_body}
}}
"#,
                function_body = Self::render_block(&body),
            )
        } else {
            panic!("Found non-`Function` `Statement` for executive function.")
        }
    }

    fn render_exec(statement: &Statement) -> String {
        if let Statement::Function { ident, args, body } = &statement {
            let mut inputs_arrays = args
                .iter()
                .filter(|arg| matches!(arg.type_, Type::ArrayRef(_)))
                .collect::<Vec<_>>();

            let inputs_dims = args
                .iter()
                .filter(|arg| matches!(arg.type_, Type::Int(_)))
                .collect::<Vec<_>>();

            let _ = inputs_arrays.pop(); // TODO: What if there are no inputs?

            let n_input_arrays = inputs_arrays.len();
            let input_shape_vecs_string = (0..n_input_arrays)
                .map(|ind| format!(
                    "let d{ind} = std::slice::from_raw_parts(inputs[{ind}].shape, inputs[{ind}].ndim);"
                ))
                .collect::<Vec<_>>()
                .join("\n");

            let input_arrays_string = (0..inputs_arrays.len())
                .map(|ind| format!(
                    "let in{ind} = std::slice::from_raw_parts(inputs[{ind}].data, d{ind}.iter().product());"
                ))
                .collect::<Vec<_>>()
                .join("\n");

            // map bound idents (i.e., `b0`, `b1`, ...) to `d0[0]`, `d1[0]`, etc.
            // that is: `d{array_ind}[{bound_ind}]`
            let mut bound_variable_string = String::new();
            let mut array_arg_ind = 0;
            let mut array_dim_ind = 0;
            let mut dim_map: HashMap<String, String> = HashMap::new();
            for arg in args {
                match arg.type_ {
                    Type::ArrayRef(_) => {
                        array_arg_ind += 1;
                        array_dim_ind = 0;
                    }
                    Type::Int(_) => {
                        array_dim_ind += 1;
                        let shape_vec_string = if array_arg_ind > n_input_arrays {
                            "out".to_string()
                        } else {
                            format!("d{}", array_arg_ind - 1)
                        };
                        let Expr::Ident(ref bound_ident) = arg.ident else {
                            panic!("")
                        };
                        bound_variable_string.push_str(&format!(
                            "let {bound_ident} = {shape_vec_string}[{}];",
                            array_dim_ind - 1
                        ))
                    }
                    _ => panic!("Unexpected arg type in exec function."),
                }
            }

            let output_shape_vec_string =
                "let dout = std::slice::from_raw_parts(output.shape, output.ndim);";

            let output_array_string =
                "let out = std::slice::from_raw_parts_mut(output.data, dout.iter().product());";

            format!(
                r#"
#[no_mangle]
unsafe extern "C"
fn f(inputs: *const Tensor, n_inputs: usize, output: *mut TensorMut) {{
    let inputs = std::slice::from_raw_parts(inputs, n_inputs);
    let output = &mut *output;

    {input_shape_vecs_string}
    {output_shape_vec_string}

    {input_arrays_string}
    {output_array_string}

    {bound_variable_string}

    {function_body}
}}
"#,
                function_body = Self::render_block(&body),
            )
        } else {
            panic!("Found non-`Function` `Statement` for executive function.")
        }
    }

    fn render_statement(statement: &Statement) -> String {
        match statement {
            Statement::Assignment { left, right } => format!(
                "{} = {};",
                Self::render_expr(left),
                Self::render_expr(right)
            ),
            Statement::Declaration {
                ident,
                value,
                type_,
            } => {
                let (Type::Int(mutable) | Type::Array(mutable) | Type::ArrayRef(mutable)) = type_;
                format!(
                    "let {}{ident}: {} = {};",
                    if *mutable { "mut " } else { "" },
                    Self::render_type(type_),
                    Self::render_expr(value)
                )
            }
            Statement::Skip { index, bound } => format!("if {index} >= {bound} {{ continue; }}"),
            Statement::Loop {
                index, bound, body, ..
            } => {
                format!(
                    "for {index} in 0..{} {{ {} }}",
                    Self::render_expr(&bound),
                    Self::render_block(body)
                )
            }
            Statement::Function { ident, args, body } => format!(
                "#[no_mangle]\nfn {ident}({}) {{{}}}",
                args.iter()
                    .map(|Arg { type_, ident }| {
                        let (Type::Int(_) | Type::Array(_) | Type::ArrayRef(_)) = type_;
                        format!("{}: {}", Self::render_expr(ident), Self::render_type(type_),)
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                Self::render_block(&body),
            ),
            Statement::Return { value } => Self::render_expr(&value),
            Statement::Call { ident, args } => format!(
                "{ident}({});",
                args.iter()
                    .map(|Arg { ident, .. }| Self::render_expr(&ident))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        }
    }
}

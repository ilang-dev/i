use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
};

use crate::{
    backend::Render,
    block::{Arg, Block, Expr, Program, Statement, Type},
};

pub struct CudaBackend;

#[derive(Clone, Debug)]
struct Dim3(String, String, String);

impl Default for Dim3 {
    fn default() -> Self {
        Dim3("1".to_string(), "1".to_string(), "1".to_string())
    }
}

impl Display for Dim3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{},{}", self.0, self.1, self.2)
    }
}

#[derive(Debug, Clone)]
struct IdentMap {
    index: String,
    dim: Dim,
}

#[derive(Clone, Debug, Default)]
struct Kernel {
    ident: String,
    args: Vec<Arg>,
    grid: Dim3,
    block: Dim3,
    statements: Vec<Statement>,
    next_dim: usize,
    ident_maps: Vec<IdentMap>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Dim {
    OuterX,
    OuterY,
    OuterZ,
    InnerX,
    InnerY,
    InnerZ,
}

impl Dim {
    fn to_index(&self) -> String {
        match &self {
            Dim::OuterX => "blockIdx.x".to_string(),
            Dim::OuterY => "blockIdx.y".to_string(),
            Dim::OuterZ => "blockIdx.z".to_string(),
            Dim::InnerX => "threadIdx.x".to_string(),
            Dim::InnerY => "threadIdx.y".to_string(),
            Dim::InnerZ => "threadIdx.z".to_string(),
        }
    }
}

impl Kernel {
    fn process(&mut self) {
        if self.statements.is_empty() {
            return;
        }
        let mut stack = VecDeque::new();
        stack.extend(self.statements.clone());
        self.statements.clear();
        while !stack.is_empty() {
            let statement = stack.pop_front().unwrap();
            if let Statement::Loop {
                ref index,
                ref bound,
                ref body,
                parallel,
            } = statement
            {
                if parallel {
                    if self.set_next_dim(index.to_string(), bound).is_err() {
                        panic!("Ran out of Kernel dimensions");
                    }
                    stack.extend(body.statements.clone());
                } else {
                    self.statements.push(statement);
                }
            } else {
                self.statements.push(statement);
            }
        }
    }
    fn render_definition(&self) -> String {
        let mut output = String::new();
        let Kernel {
            ident,
            args,
            statements,
            ident_maps,
            ..
        } = self;
        let params = args
            .iter()
            .map(CudaBackend::render_param)
            .collect::<Vec<String>>()
            .join(",");
        output += &format!("__global__ void d_{ident}({params}) {{");
        for IdentMap { index, dim, .. } in ident_maps.iter() {
            output += &format!("int {index} = {};", dim.to_index());
        }
        for statement in statements {
            output += &CudaBackend::render_statement(statement);
        }
        output += "}";
        output
    }
    fn render_call_site(&self) -> String {
        let Kernel {
            ident,
            grid,
            block,
            args,
            ..
        } = self;
        let mut output = String::new();
        output += &format!("dim3 {ident}_grid({grid});");
        output += &format!("dim3 {ident}_block({block});");
        let rendered_args = args
            .iter()
            .map(CudaBackend::render_arg)
            .collect::<Vec<String>>()
            .join(",");
        output += &format!("d_{ident}<<<{ident}_grid, {ident}_block>>>({rendered_args});");
        output += "err = cudaGetLastError();";
        output += &format!("if (err != cudaSuccess) {{fprintf(stderr, \"kernel d_{ident} failed: %s\\n\", cudaGetErrorString(err));}}");
        output += "cudaDeviceSynchronize();";
        output
    }
    fn set_next_dim(&mut self, index: String, bound: &Expr) -> Result<(), ()> {
        match self.next_dim {
            0 => {
                self.grid.0 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::OuterX,
                });
            }
            1 => {
                self.grid.1 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::OuterY,
                });
            }
            2 => {
                self.grid.2 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::OuterZ,
                });
            }
            3 => {
                self.block.0 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::InnerX,
                });
            }
            4 => {
                self.block.1 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::InnerY,
                });
            }
            5 => {
                self.block.2 = CudaBackend::render_expr(bound);
                self.ident_maps.push(IdentMap {
                    index,
                    dim: Dim::InnerZ,
                });
            }
            _ => return Err(()),
        }
        self.next_dim += 1;
        Ok(())
    }
}

impl Render for CudaBackend {
    fn render(program: &Program) -> String {
        let mut output = "#include <cuda.h>\n#include <stdio.h>\n\n".to_string();

        let mut kernels = Vec::new();

        for statement in program.library.statements.iter() {
            if let Statement::Function { ident, args, body } = statement {
                // Assume that this Function will benefit from parallel computation; might not
                // always be true or optimal
                let mut kernel = Kernel {
                    ident: ident.to_string(),
                    args: args.to_vec(),
                    statements: body.statements.clone(),
                    ..Default::default()
                };
                kernel.process();
                output += &kernel.render_definition();
                kernels.push(kernel);
            } else {
                panic!(
                    "Found non-Function Statement in root block: {:?}",
                    statement
                );
            }
        }

        let last_statement = &program.exec;
        if let Statement::Function { ident, args, body } = last_statement {
            let params = args
                .iter()
                .map(CudaBackend::render_param)
                .collect::<Vec<String>>()
                .join(",");
            output += &format!("int {ident}({params}) {{cudaError_t err;");
            for statement in body.statements.iter() {
                match statement {
                    Statement::Call { ident, .. } => {
                        if let Some(kernel) = kernels.iter().find(|k| k.ident == *ident) {
                            // TODO omit cudaDeviceSynchronize() for last kernel
                            output += &kernel.render_call_site();
                        } else {
                            output += &CudaBackend::render_statement(statement);
                        }
                    }
                    _ => output += &CudaBackend::render_statement(statement),
                }
            }
            output += "return 0;}";
        } else {
            panic!(
                "Found non-Function Statement in root block: {:?}",
                last_statement
            );
        }

        output
    }
}

impl CudaBackend {
    fn render_arg(arg: &Arg) -> String {
        let Arg { ident, .. } = arg;
        format!("{}", CudaBackend::render_expr(ident))
    }
    fn render_param(arg: &Arg) -> String {
        let Arg { type_, ident } = arg;
        format!(
            "{} {}",
            CudaBackend::render_type(type_),
            CudaBackend::render_expr(ident)
        )
    }
    fn render_type(type_: &Type) -> String {
        match type_ {
            Type::Int(_) => "int".to_string(),
            Type::Scalar(_) => "float".to_string(),
            Type::Array(_) | Type::ArrayRef(_) => "float*".to_string(),
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
                ..
            } => {
                if let Expr::Alloc {
                    initial_value: _, // TODO handle initial values?
                    shape,
                } = value
                {
                    // TODO maybe declaration and allocation should be separate
                    let shape_str = shape.join("*");
                    let mut output = format!("float *{ident};cudaMalloc(&{ident},{shape_str}*4);");
                    output += "err = cudaGetLastError();";
                    output += &format!("if (err != cudaSuccess) {{fprintf(stderr, \"cudaMalloc for {ident} failed: %s\\n\", cudaGetErrorString(err));}}");
                    output
                } else {
                    let rendered_type = CudaBackend::render_type(type_);
                    let rendered_value = CudaBackend::render_expr(value);
                    format!("{rendered_type} {ident} = {rendered_value};")
                }
            }
            Statement::Skip { index, bound } => format!("if ({index} >= {bound}) {{ continue; }}"),
            Statement::Loop {
                index, bound, body, ..
            } => format!(
                "for (int {index} = 0; {index} < {}; {index}++) {{{}}}",
                Self::render_expr(bound),
                body.statements
                    .iter()
                    .map(CudaBackend::render_statement)
                    .collect::<Vec<String>>()
                    .join("")
            ),
            Statement::Return { .. } => todo!(),
            Statement::Function { .. } => unreachable!("Reached a Function inside a Function"),
            Statement::Call { ident, args } => {
                let args = args
                    .iter()
                    .map(CudaBackend::render_arg)
                    .collect::<Vec<String>>()
                    .join(",");
                format!("{ident}({args});")
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
                    "{} > 0. ? {} : 0.",
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[0]),
                ),
                2 => format!(
                    "{} > {} ? {} : {}",
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[1]),
                    Self::render_expr(&inputs[0]),
                    Self::render_expr(&inputs[1]),
                ),
                _ => panic!("Expected 1 or 2 inputs to op `>`."),
            },
            _ => format!(
                "({})",
                inputs
                    .iter()
                    .map(|input| Self::render_expr(&input))
                    .collect::<Vec<_>>()
                    .join(&format!(" {op} "))
            ),
        }
    }
    fn render_expr(expr: &Expr) -> String {
        match expr {
            Expr::Ident(s) => s.to_string(),
            Expr::Int(x) => format!("{x}"),
            Expr::Scalar(x) => match x {
                x if x.is_nan() => "NAN".to_string(),
                x if *x == f32::INFINITY => "INFINITY".to_string(),
                x if *x == f32::NEG_INFINITY => "-INFINITY".to_string(),
                x => {
                    let s = format!("{:.8}", x);
                    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
                    let with_decimal = if trimmed.contains('.') {
                        trimmed.to_string()
                    } else {
                        format!("{}.", trimmed)
                    };
                    format!("{}f", with_decimal)
                }
            },
            Expr::Op { .. } => Self::render_op(&expr),
            Expr::Indexed { ident, index } => format!("{ident}[{}]", Self::render_expr(&index)),
            Expr::Alloc { .. } => {
                unreachable!("Expr::Alloc should be handled in Statement::Declaration")
            }
            Expr::Ref(ident, _) => ident.to_string(),
        }
    }
}

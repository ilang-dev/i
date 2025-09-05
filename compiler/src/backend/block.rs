use crate::backend::Render;
use crate::block::{Arg, Block, Expr, Program, Statement, Type};

pub struct BlockBackend;

impl Render for BlockBackend {
    fn render(program: &Program) -> String {
        format!(
            "{}\n{}",
            Self::render_block(&program.library, 0),
            Self::render_statement(&program.exec, 0)
        )
    }
}

impl BlockBackend {
    fn indent(level: usize) -> String {
        "  ".repeat(level)
    }

    fn render_block(block: &Block, level: usize) -> String {
        let stmts = block
            .statements
            .iter()
            .map(|s| Self::render_statement(s, level + 1))
            .collect::<Vec<_>>()
            .join("\n");
        format!("(\n{}\n{})", stmts, Self::indent(level))
    }

    fn render_type(t: &Type) -> String {
        match t {
            Type::Int(false) => "i".into(),
            Type::Int(true) => "i!".into(),
            Type::Scalar(false) => "s".into(),
            Type::Scalar(true) => "s!".into(),
            Type::Array(false) => "a".into(),
            Type::Array(true) => "a!".into(),
            Type::ArrayRef(false) => "ar".into(),
            Type::ArrayRef(true) => "ar!".into(),
        }
    }

    fn render_expr(expr: &Expr) -> String {
        match expr {
            Expr::Alloc {
                initial_value,
                shape,
            } => format!(
                "(alloc {} {})",
                Self::render_expr(&initial_value),
                shape.join(" ")
            ),
            Expr::Int(x) => format!("(int {x})"),
            Expr::Scalar(x) => format!("(scal {x})"),
            Expr::Ident(s) => format!("(id {s})"),
            Expr::Ref(s, true) => format!("(ref! {s})"),
            Expr::Ref(s, false) => format!("(ref {s})"),
            Expr::Op { op, inputs } => {
                let parts = inputs
                    .iter()
                    .map(Self::render_expr)
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("(op {} {})", op, parts)
            }
            Expr::Indexed { ident, index } => {
                format!("(index {ident} {})", Self::render_expr(index))
            }
        }
    }

    fn render_statement(stmt: &Statement, level: usize) -> String {
        let ind = Self::indent(level);
        match stmt {
            Statement::Assignment { left, right } => {
                format!(
                    "{}(assign\n{}{}\n{}{})",
                    ind,
                    Self::indent(level + 1),
                    Self::render_expr(left),
                    Self::indent(level + 1),
                    Self::render_expr(right)
                )
            }
            Statement::Declaration {
                ident,
                value,
                type_,
            } => {
                format!(
                    "{}(decl {} {} {})",
                    ind,
                    ident,
                    Self::render_type(type_),
                    Self::render_expr(value)
                )
            }
            Statement::Skip { index, bound } => format!("{}(skip {} {})", ind, index, bound),
            Statement::Loop {
                index,
                bound,
                body,
                parallel,
            } => {
                let p = if *parallel { "1" } else { "0" };
                format!(
                    "{}(loop {} {} {} {})",
                    ind,
                    index,
                    Self::render_expr(bound),
                    p,
                    Self::render_block(body, level + 1)
                )
            }
            Statement::Return { value } => format!("{}(return {})", ind, Self::render_expr(value)),
            Statement::Function { ident, args, body } => {
                let rendered_args = args
                    .iter()
                    .map(|Arg { type_, ident }| {
                        format!(
                            "\n{}(arg {} {})",
                            Self::indent(level + 1),
                            Self::render_type(type_),
                            Self::render_expr(ident)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("");
                format!(
                    "{}(func {} ({}) {})",
                    ind,
                    ident,
                    rendered_args,
                    Self::render_block(body, level + 1)
                )
            }
            Statement::Call { ident, args } => {
                let rendered_args = args
                    .iter()
                    .map(|Arg { type_, ident }| {
                        format!(
                            "(arg {} {})",
                            Self::render_type(type_),
                            Self::render_expr(ident)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{}(call {} ({}))", ind, ident, rendered_args)
            }
        }
    }
}

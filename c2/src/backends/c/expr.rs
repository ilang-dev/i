use crate::ir::common::Op;
use crate::ir::module::{Cast, Expr, Field, Place};

use super::fmt::indent;
use super::render_ident;

pub(super) fn render_expr(expr: &Expr) -> String {
    render_expr_in(expr, Context::default()).source
}

pub(super) fn render_place(place: &Place) -> String {
    render_place_raw(place).source
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl Operator {
    fn symbol(self) -> &'static str {
        match self {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
            Operator::Rem => "%",
            Operator::Lt => "<",
            Operator::Gt => ">",
            Operator::Le => "<=",
            Operator::Ge => ">=",
            Operator::Eq => "==",
            Operator::Ne => "!=",
            Operator::And => "&&",
            Operator::Or => "||",
        }
    }

    fn precedence(self) -> u8 {
        match self {
            Operator::Mul | Operator::Div | Operator::Rem => 70,
            Operator::Add | Operator::Sub => 60,
            Operator::Lt | Operator::Gt | Operator::Le | Operator::Ge => 50,
            Operator::Eq | Operator::Ne => 45,
            Operator::And => 40,
            Operator::Or => 30,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Context {
    parent: Option<Operator>,
    side: Option<Side>,
}

#[derive(Clone, Debug)]
struct Rendered {
    source: String,
    precedence: u8,
    operator: Option<Operator>,
}

impl Rendered {
    fn primary(source: String) -> Self {
        Self {
            source,
            precedence: 100,
            operator: None,
        }
    }
}

fn render_expr_in(expr: &Expr, context: Context) -> Rendered {
    let rendered = match expr {
        Expr::Usize(value) => Rendered::primary(value.to_string()),
        Expr::Scalar(value) => Rendered::primary(render_scalar(*value)),
        Expr::Ident(ident) => Rendered::primary(render_ident(ident)),
        Expr::Index { base, index } => render_index_expr(base, index),
        Expr::Field { base, field } => {
            let base = render_expr_in(base, Context::default());
            Rendered::primary(format!("{}.{}", base.source, render_field(*field)))
        }
        Expr::Cast { kind, value } => Rendered::primary(render_cast(*kind, value)),
        Expr::Op { op, args } => return render_op(*op, args, context),
        Expr::Add(lhs, rhs) => render_binary(Operator::Add, lhs, rhs),
        Expr::Sub(lhs, rhs) => render_binary(Operator::Sub, lhs, rhs),
        Expr::Mul(lhs, rhs) => render_binary(Operator::Mul, lhs, rhs),
        Expr::Div(lhs, rhs) => render_binary(Operator::Div, lhs, rhs),
        Expr::Rem(lhs, rhs) => render_binary(Operator::Rem, lhs, rhs),
        Expr::Lt(lhs, rhs) => render_binary(Operator::Lt, lhs, rhs),
        Expr::Eq(lhs, rhs) => render_binary(Operator::Eq, lhs, rhs),
        Expr::And(lhs, rhs) => render_binary(Operator::And, lhs, rhs),
    };
    apply_context(rendered, context)
}

fn render_place_raw(place: &Place) -> Rendered {
    match place {
        Place::Ident(ident) => Rendered::primary(render_ident(ident)),
        Place::Index { base, index } => render_index_place(base, index),
        Place::Field { base, field } => {
            let base = render_place_raw(base);
            Rendered::primary(format!("{}.{}", base.source, render_field(*field)))
        }
    }
}

fn render_op(op: Op, args: &[Expr], context: Context) -> Rendered {
    match op {
        Op::Add => render_joined(Operator::Add, args, context),
        Op::Mul => render_joined(Operator::Mul, args, context),
        Op::Div => render_joined(Operator::Div, args, context),
        Op::Sub => render_joined(Operator::Sub, args, context),
        Op::Max => Rendered::primary(render_call_fold("fmaxf", args)),
        Op::Min => Rendered::primary(render_call_fold("fminf", args)),
        Op::Pow => Rendered::primary(render_call_fold("powf", args)),
        Op::Log => Rendered::primary(render_unary_call("logf", args)),
        Op::Gt => render_joined(Operator::Gt, args, context),
        Op::Ge => render_joined(Operator::Ge, args, context),
        Op::Lt => render_joined(Operator::Lt, args, context),
        Op::Le => render_joined(Operator::Le, args, context),
        Op::Eq => render_joined(Operator::Eq, args, context),
        Op::Ne => render_joined(Operator::Ne, args, context),
        Op::And => render_joined(Operator::And, args, context),
        Op::Or => render_joined(Operator::Or, args, context),
        Op::Xor => Rendered::primary(render_xor(args)),
        Op::Not => render_unary("!", &args[0], context),
    }
}

fn render_cast(kind: Cast, value: &Expr) -> String {
    match kind {
        Cast::View => format!("as_view(&{})", render_expr(value)),
        Cast::ViewMut => format!("as_view_mut(&{})", render_expr(value)),
        Cast::ReadOnly => format!("view_mut_as_view(&{})", render_expr(value)),
    }
}

fn render_index_expr(base: &Expr, index: &Expr) -> Rendered {
    let multiline_affine = expr_is_data_field(base);
    let base = render_expr_in(base, Context::default());
    Rendered::primary(render_index(base.source, index, multiline_affine))
}

fn render_index_place(base: &Place, index: &Expr) -> Rendered {
    let multiline_affine = place_is_data_field(base);
    let base = render_place_raw(base);
    Rendered::primary(render_index(base.source, index, multiline_affine))
}

fn render_index(base: String, index: &Expr, multiline_affine: bool) -> String {
    if multiline_affine && is_affine_sum(index) {
        format!("{base}[\n{}\n]", indent(&render_affine_index(index)))
    } else {
        format!(
            "{base}[{}]",
            render_expr_in(index, Context::default()).source
        )
    }
}

fn expr_is_data_field(base: &Expr) -> bool {
    matches!(
        base,
        Expr::Field {
            field: Field::Data,
            ..
        }
    )
}

fn place_is_data_field(base: &Place) -> bool {
    matches!(
        base,
        Place::Field {
            field: Field::Data,
            ..
        }
    )
}

fn render_binary(operator: Operator, lhs: &Expr, rhs: &Expr) -> Rendered {
    let lhs = render_child(lhs, operator, Side::Left);
    let rhs = render_child(rhs, operator, Side::Right);
    Rendered {
        source: format!("{} {} {}", lhs.source, operator.symbol(), rhs.source),
        precedence: operator.precedence(),
        operator: Some(operator),
    }
}

fn render_joined(operator: Operator, args: &[Expr], context: Context) -> Rendered {
    let Some((first, rest)) = args.split_first() else {
        return Rendered::primary("0.0f".to_string());
    };

    let mut source = render_child(first, operator, Side::Left).source;
    for arg in rest {
        let arg = render_child(arg, operator, Side::Right);
        source.push_str(&format!(" {} {}", operator.symbol(), arg.source));
    }

    apply_context(
        Rendered {
            source,
            precedence: operator.precedence(),
            operator: Some(operator),
        },
        context,
    )
}

fn render_child(expr: &Expr, parent: Operator, side: Side) -> Rendered {
    render_expr_in(
        expr,
        Context {
            parent: Some(parent),
            side: Some(side),
        },
    )
}

fn render_unary(operator: &str, value: &Expr, context: Context) -> Rendered {
    let value = render_expr_in(value, Context::default());
    let rendered = Rendered {
        source: format!(
            "{operator}{}",
            parenthesize_if(value.source, value.precedence < 80)
        ),
        precedence: 80,
        operator: None,
    };
    apply_context(rendered, context)
}

fn apply_context(rendered: Rendered, context: Context) -> Rendered {
    let Some(parent) = context.parent else {
        return rendered;
    };
    let needs_parens = rendered.precedence < parent.precedence()
        || (rendered.precedence == parent.precedence()
            && context.side == Some(Side::Right)
            && !right_child_can_drop_parens(parent, rendered.operator));
    Rendered {
        source: parenthesize_if(rendered.source, needs_parens),
        ..rendered
    }
}

fn right_child_can_drop_parens(parent: Operator, child: Option<Operator>) -> bool {
    matches!(
        (parent, child),
        (Operator::Add, Some(Operator::Add))
            | (Operator::Mul, Some(Operator::Mul))
            | (Operator::And, Some(Operator::And))
            | (Operator::Or, Some(Operator::Or))
    )
}

fn parenthesize_if(source: String, condition: bool) -> String {
    if condition {
        format!("({source})")
    } else {
        source
    }
}

fn is_affine_sum(expr: &Expr) -> bool {
    matches!(expr, Expr::Add(_, _))
}

fn render_affine_index(expr: &Expr) -> String {
    let mut terms = Vec::new();
    collect_add_terms(expr, &mut terms);
    terms
        .iter()
        .enumerate()
        .map(|(index, term)| {
            let rendered = render_expr_in(term, Context::default()).source;
            if index + 1 == terms.len() {
                rendered
            } else {
                format!("{rendered} +")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_add_terms<'a>(expr: &'a Expr, terms: &mut Vec<&'a Expr>) {
    match expr {
        Expr::Add(lhs, rhs) => {
            collect_add_terms(lhs, terms);
            collect_add_terms(rhs, terms);
        }
        _ => terms.push(expr),
    }
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

fn render_field(field: Field) -> &'static str {
    match field {
        Field::Data => "data",
        Field::Shape => "shape",
        Field::Layout => "layout",
        Field::Rank => "rank",
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

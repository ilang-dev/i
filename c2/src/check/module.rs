use std::collections::BTreeSet;
use std::fmt;

use crate::ir::module::{Block, Expr, Fn, Ident, Module, Place, Signature, Stmt};

pub fn validate_module(module: &Module) -> Result<(), ValidationError> {
    validate_signature(&module.count, Signature::Count, "count")?;
    validate_signature(&module.ranks, Signature::Ranks, "ranks")?;
    validate_signature(&module.shapes, Signature::Shapes, "shapes")?;
    validate_signature(&module.exec, Signature::Exec, "exec")?;
    for (kernel_index, kernel) in module.kernels.iter().enumerate() {
        validate_signature(
            kernel,
            Signature::Kernel,
            &format!("kernel {}", kernel_index),
        )?;
    }

    let mut functions = BTreeSet::new();
    insert_function(&mut functions, &module.count.ident)?;
    insert_function(&mut functions, &module.ranks.ident)?;
    insert_function(&mut functions, &module.shapes.ident)?;
    insert_function(&mut functions, &module.exec.ident)?;
    for kernel in &module.kernels {
        insert_function(&mut functions, &kernel.ident)?;
    }

    let kernels = module
        .kernels
        .iter()
        .map(|kernel| kernel.ident.0.as_str())
        .collect::<BTreeSet<_>>();

    validate_function(&module.count, &kernels)?;
    validate_function(&module.ranks, &kernels)?;
    validate_function(&module.shapes, &kernels)?;
    validate_function(&module.exec, &kernels)?;
    for kernel in &module.kernels {
        validate_function(kernel, &kernels)?;
    }

    Ok(())
}

fn validate_signature(
    function: &Fn,
    expected: Signature,
    name: &str,
) -> Result<(), ValidationError> {
    if function.signature != expected {
        return Err(err(format!(
            "{} has signature {:?}",
            name, function.signature
        )));
    }
    Ok(())
}

fn insert_function(functions: &mut BTreeSet<String>, ident: &Ident) -> Result<(), ValidationError> {
    if !functions.insert(ident.0.clone()) {
        return Err(err(format!("function {} is repeated", ident.0)));
    }
    Ok(())
}

fn validate_function(function: &Fn, kernels: &BTreeSet<&str>) -> Result<(), ValidationError> {
    let mut scope = implicit_scope(function.signature);
    validate_block(function.signature, kernels, &mut scope, &function.body)
        .map_err(|message| err(format!("function {} {}", function.ident.0, message)))
}

fn implicit_scope(signature: Signature) -> BTreeSet<String> {
    match signature {
        Signature::Count => BTreeSet::new(),
        Signature::Ranks => ["ranks"].into_iter().map(str::to_string).collect(),
        Signature::Shapes => ["inputs", "shapes"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Signature::Exec => ["inputs", "outputs"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Signature::Kernel => ["readonlys", "writeables"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    }
}

fn validate_block(
    signature: Signature,
    kernels: &BTreeSet<&str>,
    scope: &mut BTreeSet<String>,
    block: &Block,
) -> Result<(), String> {
    for statement in &block.0 {
        match statement {
            Stmt::Let { ident, value, .. } => {
                validate_expr(scope, value)?;
                bind(scope, ident)?;
            }
            Stmt::Set { dst, value } => {
                validate_place(scope, dst)?;
                validate_expr(scope, value)?;
            }
            Stmt::Alloc { dst, shape, layout } => {
                if signature != Signature::Exec {
                    return Err("alloc appears outside exec".to_string());
                }
                for expr in shape.iter().chain(layout) {
                    validate_expr(scope, expr)?;
                }
                bind(scope, dst)?;
            }
            Stmt::Free(ident) => {
                if signature != Signature::Exec {
                    return Err("free appears outside exec".to_string());
                }
                validate_ident(scope, ident)?;
            }
            Stmt::Dispatch {
                kernel,
                reads,
                writes,
            } => {
                if signature != Signature::Exec {
                    return Err("dispatch appears outside exec".to_string());
                }
                if !kernels.contains(kernel.0.as_str()) {
                    return Err(format!(
                        "dispatch references nonexistent kernel {}",
                        kernel.0
                    ));
                }
                for ident in reads.iter().chain(writes) {
                    validate_ident(scope, ident)?;
                }
            }
            Stmt::Loop { iter, bound, body } => {
                validate_expr(scope, bound)?;
                let mut child = scope.clone();
                bind(&mut child, iter)?;
                validate_block(signature, kernels, &mut child, body)?;
            }
            Stmt::If { cond, body } => {
                validate_expr(scope, cond)?;
                let mut child = scope.clone();
                validate_block(signature, kernels, &mut child, body)?;
            }
            Stmt::Return(value) => match (signature, value) {
                (Signature::Count, Some(expr)) => validate_expr(scope, expr)?,
                (Signature::Count, None) => {
                    return Err("count returns without value".to_string());
                }
                (_, Some(_)) => return Err("void function returns a value".to_string()),
                (_, None) => {}
            },
        }
    }
    Ok(())
}

fn bind(scope: &mut BTreeSet<String>, ident: &Ident) -> Result<(), String> {
    if !scope.insert(ident.0.clone()) {
        return Err(format!("ident {} is repeated", ident.0));
    }
    Ok(())
}

fn validate_ident(scope: &BTreeSet<String>, ident: &Ident) -> Result<(), String> {
    if !scope.contains(&ident.0) {
        return Err(format!("references unbound ident {}", ident.0));
    }
    Ok(())
}

fn validate_expr(scope: &BTreeSet<String>, expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Usize(_) | Expr::Scalar(_) => Ok(()),
        Expr::Ident(ident) => validate_ident(scope, ident),
        Expr::Index { base, index } => {
            validate_expr(scope, base)?;
            validate_expr(scope, index)
        }
        Expr::Field { base, .. } => validate_expr(scope, base),
        Expr::Cast { value, .. } => validate_expr(scope, value),
        Expr::Op { args, .. } => {
            for arg in args {
                validate_expr(scope, arg)?;
            }
            Ok(())
        }
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Rem(lhs, rhs)
        | Expr::Lt(lhs, rhs) => {
            validate_expr(scope, lhs)?;
            validate_expr(scope, rhs)
        }
    }
}

fn validate_place(scope: &BTreeSet<String>, place: &Place) -> Result<(), String> {
    match place {
        Place::Ident(ident) => validate_ident(scope, ident),
        Place::Index { base, index } => {
            validate_place(scope, base)?;
            validate_expr(scope, index)
        }
        Place::Field { base, .. } => validate_place(scope, base),
    }
}

fn err(message: impl Into<String>) -> ValidationError {
    ValidationError {
        message: message.into(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::validate_module;
    use crate::ir::module::{Block, Expr, Fn, Ident, Module, Place, Signature, Stmt, Type};

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

    fn module() -> Module {
        Module {
            count: function(
                "count",
                Signature::Count,
                Block(vec![Stmt::Return(Some(Expr::Usize(1)))]),
            ),
            ranks: function(
                "ranks",
                Signature::Ranks,
                Block(vec![
                    Stmt::Set {
                        dst: Place::Index {
                            base: Box::new(Place::Ident(id("ranks"))),
                            index: Expr::Usize(0),
                        },
                        value: Expr::Usize(2),
                    },
                    Stmt::Return(None),
                ]),
            ),
            shapes: function("shapes", Signature::Shapes, Block(vec![Stmt::Return(None)])),
            exec: function(
                "exec",
                Signature::Exec,
                Block(vec![
                    Stmt::Let {
                        ident: id("in0"),
                        ty: Type::View,
                        value: Expr::Ident(id("inputs")),
                    },
                    Stmt::Dispatch {
                        kernel: id("f0"),
                        reads: vec![id("in0")],
                        writes: vec![],
                    },
                    Stmt::Return(None),
                ]),
            ),
            kernels: vec![function("f0", Signature::Kernel, Block(vec![]))],
        }
    }

    #[test]
    fn accepts_module() {
        assert!(validate_module(&module()).is_ok());
    }

    #[test]
    fn rejects_bad_signature() {
        let mut module = module();
        module.count.signature = Signature::Exec;

        let error = validate_module(&module).unwrap_err();
        assert_eq!(error.to_string(), "count has signature Exec");
    }

    #[test]
    fn rejects_duplicate_function_ident() {
        let mut module = module();
        module.kernels[0].ident = id("exec");

        let error = validate_module(&module).unwrap_err();
        assert_eq!(error.to_string(), "function exec is repeated");
    }

    #[test]
    fn rejects_unbound_ident() {
        let mut module = module();
        module.exec.body.0.insert(
            0,
            Stmt::Set {
                dst: Place::Ident(id("missing")),
                value: Expr::Usize(0),
            },
        );

        let error = validate_module(&module).unwrap_err();
        assert_eq!(
            error.to_string(),
            "function exec references unbound ident missing"
        );
    }

    #[test]
    fn rejects_unknown_dispatch_kernel() {
        let mut module = module();
        let Stmt::Dispatch { kernel, .. } = &mut module.exec.body.0[1] else {
            unreachable!();
        };
        *kernel = id("f1");

        let error = validate_module(&module).unwrap_err();
        assert_eq!(
            error.to_string(),
            "function exec dispatch references nonexistent kernel f1"
        );
    }

    #[test]
    fn rejects_dispatch_outside_exec() {
        let mut module = module();
        module.kernels[0].body = Block(vec![Stmt::Dispatch {
            kernel: id("f0"),
            reads: vec![],
            writes: vec![],
        }]);

        let error = validate_module(&module).unwrap_err();
        assert_eq!(
            error.to_string(),
            "function f0 dispatch appears outside exec"
        );
    }

    #[test]
    fn rejects_value_return_from_void_function() {
        let mut module = module();
        module.exec.body = Block(vec![Stmt::Return(Some(Expr::Usize(0)))]);

        let error = validate_module(&module).unwrap_err();
        assert_eq!(
            error.to_string(),
            "function exec void function returns a value"
        );
    }
}

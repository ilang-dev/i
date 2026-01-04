#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Int(usize),
    Scalar(f32),
    Ident(String),
    Op {
        op: char,
        inputs: Vec<Expr>,
    },
    Indexed {
        expr: Box<Expr>, // Should be of variant `Expr::Ident` or `Expr::Indexed`
        index: Box<Expr>,
    },
    ShapeOf(Box<Expr>),
    DataOf(Box<Expr>),
    ReadOnly(Box<Expr>), // for casting `TensorMut` to `Tensor`
}

#[derive(Clone, Debug)]
pub enum Type {
    Int(bool),
    Scalar(bool),
}

#[derive(Clone, Debug)]
pub enum FunctionSignature {
    Count,
    Ranks,
    Shapes,
    Exec,
    Kernel(Expr), // Inner `Expr` must be of variant `Ident`
}

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment {
        left: Expr, // Should LValue become it's own enum?
        right: Expr,
    },
    Alloc {
        index: usize,
        initial_value: Box<Expr>, // must be of variant `Scalar`
        shape: Vec<Expr>,
    },
    Declaration {
        ident: Expr, // must be an "LValue"
        value: Expr,
        type_: Type,
    },
    Skip {
        index: Expr,
        bound: Expr,
    },
    Loop {
        index: Expr,
        bound: Expr,
        body: Block,
        parallel: bool,
    },
    Return {
        value: Expr,
    },
    Function {
        signature: FunctionSignature,
        body: Block,
    },
    Call {
        ident: Expr,         // must be of variant `Expr::Ident`
        in_args: Vec<Expr>,  // must be of variant `Expr::Ident`
        out_args: Vec<Expr>, // must be of variant `Expr::Ident`
    }, // This is a Statement because it's only ever used as one
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub count: Statement,  // Should be `Statement::Function`
    pub ranks: Statement,  // Should be `Statement::Function`
    pub shapes: Statement, // Should be `Statement::Function`
    pub library: Block,    // Should consist only of `Statement::Function`s
    pub exec: Statement,   // Should be `Statement::Function`
}

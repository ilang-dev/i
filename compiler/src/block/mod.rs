#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Alloc {
        initial_value: Box<Expr>, // must be of variant `Scalar`
        shape: Vec<Expr>,
    },
    Int(usize),
    Scalar(f32),
    Ident(String),
    Ref(String, bool), // like Ident(_), but a ref (and tracks mutability)
    Op {
        op: char,
        inputs: Vec<Expr>,
    },
    Indexed {
        expr: Box<Expr>, // Should be of variant `Expr::Ident` or `Expr::Indexed`
        index: Box<Expr>,
    },
}

// Should this be an Expr variant?
#[derive(Clone, Debug)]
pub struct Arg {
    pub type_: Type,
    pub ident: Expr, // Should be Ident(_) or a Ref(_)
}

#[derive(Clone, Debug)]
pub enum Type {
    Int(bool),
    Scalar(bool),
    Array(bool),
    ArrayRef(bool),
}

#[derive(Clone, Debug)]
pub enum FunctionSignature {
    Count,
    Ranks,
    Shapes,
    Exec,
    Kernel(String),
}

#[derive(Clone, Debug)]
pub enum Statement {
    Assignment {
        left: Expr, // Should LValue become it's own enum?
        right: Expr,
    },
    Declaration {
        ident: String,
        value: Expr,
        type_: Type,
    },
    Skip {
        // TODO: These should both probably be Expr (Ident)
        index: String,
        bound: String,
    },
    Loop {
        index: String,
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
        ident: String,
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

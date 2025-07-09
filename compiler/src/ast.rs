use std::collections::HashMap;

#[derive(Debug)]
pub struct AST(pub Vec<NamedExpr>, pub ExprRef);

#[derive(Debug)]
pub struct NamedExpr {
    pub ident: Symbol,
    pub expr_ref: ExprRef,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Index(IndexExpr),
    Combinator(Combinator),
}

/// Holds all Exprs
#[derive(Debug)]
pub struct ExprBank(pub Vec<Expr>);

/// An index into the ExprBank
#[derive(Clone, Copy, Debug)]
pub struct ExprRef(pub usize);

#[derive(Clone, Debug)]
pub struct IndexExpr {
    pub op: ScalarOp,
    pub out: Symbol,
    pub schedule: Schedule,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    // Should we have a `SplitTable` AST type? What about `Int` and using it and `Symbol` here?
    pub splits: HashMap<char, Vec<usize>>, // loop index, split factors
    // loop index, position in split list +1 (0 reserved for base loop)
    pub loop_order: Vec<(char, usize)>,
    // for each input, tracks the `loop_order` index+1 of the computation loop level? (0 reserved
    // for root level)
    pub compute_levels: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum ScalarOp {
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    NoOp(NoOp),
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Mul(Symbol, Symbol),
    Div(Symbol, Symbol),
    Add(Symbol, Symbol),
    Sub(Symbol, Symbol),
    Max(Symbol, Symbol),
    Min(Symbol, Symbol),
}

#[derive(Clone, Debug)]
pub enum UnaryOp {
    Accum(Symbol),
    Prod(Symbol),
    Neg(Symbol),
    Recip(Symbol),
    Max(Symbol),
    Min(Symbol),
    Exp(Symbol),
    Log(Symbol),
    Sqrt(Symbol),
    Abs(Symbol),
    Relu(Symbol),
}

#[derive(Clone, Debug)]
pub struct NoOp(pub Symbol);

#[derive(Clone, Debug)]
pub enum Combinator {
    Chain(ExprRef, ExprRef),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Symbol(pub String);

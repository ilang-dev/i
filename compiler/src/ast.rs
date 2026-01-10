use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Expr {
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
pub struct ScalarOp {
    pub op: Op,
    pub args: Vec<Symbol>,
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Op {
    NoOp, // unary (reshape)
    Exp, Log, Sqrt, Abs, Relu, // unary (elementwise)
    Neg, Recip, // unary, defaultable binary
    Mul, Add, Max, Min, // unary, reducing binary
    Div, Sub, // binary
}

#[derive(Clone, Debug)]
pub struct Symbol(pub String);

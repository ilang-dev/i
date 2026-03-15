use super::common::{ExprId, LoopVar, Op, Pattern, Split};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Component {
    Expr(Expr),
    Compose(Box<Component>, Box<Component>),
    Chain(Box<Component>, Box<Component>),
    Fanout(Box<Component>, Box<Component>),
    Pair(Box<Component>, Box<Component>),
    Swap(Box<Component>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expr {
    pub id: ExprId,
    pub op: Op,
    pub inputs: Vec<Pattern>,
    pub output: Pattern,
    pub schedule: Schedule,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Schedule {
    pub splits: Vec<Split>,
    pub order: Vec<LoopVar>,
    pub compute_at: Vec<Option<LoopVar>>,
}

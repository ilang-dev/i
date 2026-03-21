pub use super::expr::{Expr, PermutationAtom};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Component {
    Expr(Expr),
    Compose(Box<Component>, Box<Component>),
    Chain(Box<Component>, Box<Component>),
    Fanout(Box<Component>, Box<Component>),
    Pair(Box<Component>, Box<Component>),
    Swap(Box<Component>),
}

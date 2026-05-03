use std::ops::{BitAnd, BitOr};

use crate::ir::component::{Component, Expr};

pub fn identity() -> Component {
    Component::Identity
}

pub fn expr(expr: Expr) -> Component {
    Component::Expr(expr)
}

pub fn compose(left: Component, right: Component) -> Component {
    Component::Compose(Box::new(left), Box::new(right))
}

pub fn chain(left: Component, right: Component) -> Component {
    Component::Chain(Box::new(left), Box::new(right))
}

pub fn fanout(left: Component, right: Component) -> Component {
    Component::Fanout(Box::new(left), Box::new(right))
}

pub fn pair(left: Component, right: Component) -> Component {
    Component::Pair(Box::new(left), Box::new(right))
}

pub fn swap(component: Component) -> Component {
    Component::Swap(Box::new(component))
}

pub fn finalize(component: Component) -> Component {
    component
}

pub fn renumber_expr_ids(_component: &mut Component) {}

impl From<Expr> for Component {
    fn from(value: Expr) -> Self {
        expr(value)
    }
}

impl Component {
    pub fn identity() -> Self {
        identity()
    }

    pub fn compose(self, other: Self) -> Self {
        compose(self, other)
    }

    pub fn chain(self, other: Self) -> Self {
        chain(self, other)
    }

    pub fn fanout(self, other: Self) -> Self {
        fanout(self, other)
    }

    pub fn pair(self, other: Self) -> Self {
        pair(self, other)
    }

    pub fn swap(self) -> Self {
        swap(self)
    }

    pub fn finalize(self) -> Self {
        finalize(self)
    }
}

impl BitAnd for Component {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        fanout(self, rhs)
    }
}

impl BitOr for Component {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        chain(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::{chain, compose, expr, fanout, finalize, identity, pair, renumber_expr_ids, swap};
    use crate::ir::common::Op;
    use crate::ir::component::Component;
    use crate::ir::expr::Expr;
    #[test]
    fn lifts_expr_into_component() {
        let lifted = expr(make_expr(7, Op::Add, &["i", "i"], "i"));

        assert_eq!(
            lifted,
            Component::Expr(make_expr(7, Op::Add, &["i", "i"], "i"))
        );
    }

    #[test]
    fn builds_identity_component() {
        assert_eq!(identity(), Component::Identity);
        assert_eq!(Component::identity(), Component::Identity);
    }

    #[test]
    fn builds_each_combinator_variant() {
        let left = expr(make_expr(0, Op::Mul, &["ik", "kj"], "ijk"));
        let right = expr(make_expr(0, Op::Add, &["ijk"], "ij"));

        assert!(matches!(
            compose(left.clone(), right.clone()),
            Component::Compose(_, _)
        ));
        assert!(matches!(
            chain(left.clone(), right.clone()),
            Component::Chain(_, _)
        ));
        assert!(matches!(
            left.clone().chain(right.clone()),
            Component::Chain(_, _)
        ));
        assert!(matches!(
            left.clone() | right.clone(),
            Component::Chain(_, _)
        ));
        assert!(matches!(
            fanout(left.clone(), right.clone()),
            Component::Fanout(_, _)
        ));
        assert!(matches!(
            left.clone() & right.clone(),
            Component::Fanout(_, _)
        ));
        assert!(matches!(
            pair(left.clone(), right.clone()),
            Component::Pair(_, _)
        ));
        assert!(matches!(swap(left), Component::Swap(_)));
    }

    #[test]
    fn supports_nested_mixed_arity_construction() {
        let component = expr(make_expr(99, Op::Mul, &["ik", "kj"], "ijk"))
            .fanout(expr(make_expr(99, Op::Add, &["ij"], "i")))
            .chain(
                expr(make_expr(99, Op::Div, &["ijk", "i"], "ijk")).pair(expr(make_expr(
                    99,
                    Op::Not,
                    &["ij"],
                    "ij",
                ))),
            )
            .swap();

        assert!(matches!(component, Component::Swap(_)));
        if let Component::Swap(inner) = component {
            assert!(matches!(*inner, Component::Chain(_, _)));
        }
    }

    #[test]
    fn finalize_preserves_structure() {
        let component = expr(make_expr(10, Op::Add, &["ij"], "i")).pair(
            expr(make_expr(20, Op::Mul, &["ik", "kj"], "ij")).compose(expr(make_expr(
                30,
                Op::Add,
                &["ij"],
                "ij",
            ))),
        );

        let finalized = finalize(component);

        assert!(matches!(finalized, Component::Pair(_, _)));
    }

    #[test]
    fn renumber_expr_ids_is_a_no_op() {
        let mut component = expr(make_expr(3, Op::Add, &["i"], "i"))
            .fanout(expr(make_expr(8, Op::Mul, &["i", "i"], "i")))
            .pair(expr(make_expr(13, Op::Sub, &["i"], "i")));

        let before = component.clone();
        renumber_expr_ids(&mut component);
        assert_eq!(component, before);
    }

    fn make_expr(id: usize, op: Op, inputs: &[&str], output: &str) -> Expr {
        let _ = id;
        Expr {
            op,
            inputs: inputs
                .iter()
                .map(|pattern| pattern.chars().collect())
                .collect(),
            output: output.chars().collect(),
            splits: Vec::new(),
            permutation: Vec::new(),
        }
    }
}

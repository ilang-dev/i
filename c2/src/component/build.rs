use crate::ir::component::{Component, Expr};

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
    let mut next_expr_id = 0usize;
    renumber_component(component, &mut next_expr_id)
}

pub fn renumber_expr_ids(component: &mut Component) {
    let mut next_expr_id = 0usize;
    renumber_component_in_place(component, &mut next_expr_id);
}

impl From<Expr> for Component {
    fn from(value: Expr) -> Self {
        expr(value)
    }
}

impl Component {
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

fn renumber_component(mut component: Component, next_expr_id: &mut usize) -> Component {
    renumber_component_in_place(&mut component, next_expr_id);
    component
}

fn renumber_component_in_place(component: &mut Component, next_expr_id: &mut usize) {
    match component {
        Component::Expr(expr) => {
            expr.id = *next_expr_id;
            *next_expr_id += 1;
        }
        Component::Compose(left, right)
        | Component::Chain(left, right)
        | Component::Fanout(left, right)
        | Component::Pair(left, right) => {
            renumber_component_in_place(left, next_expr_id);
            renumber_component_in_place(right, next_expr_id);
        }
        Component::Swap(inner) => renumber_component_in_place(inner, next_expr_id),
    }
}

#[cfg(test)]
mod tests {
    use super::{chain, compose, expr, fanout, finalize, pair, renumber_expr_ids, swap};
    use crate::ir::common::{Op, Pattern};
    use crate::ir::component::{Component, Expr, Schedule};

    #[test]
    fn lifts_expr_into_component() {
        let lifted = expr(make_expr(7, Op::Add, &["i", "i"], "i"));

        assert_eq!(
            lifted,
            Component::Expr(make_expr(7, Op::Add, &["i", "i"], "i"))
        );
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
            fanout(left.clone(), right.clone()),
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
    fn finalize_renumbers_expr_ids_left_to_right() {
        let component = expr(make_expr(10, Op::Add, &["ij"], "i")).pair(
            expr(make_expr(20, Op::Mul, &["ik", "kj"], "ij")).compose(expr(make_expr(
                30,
                Op::Add,
                &["ij"],
                "ij",
            ))),
        );

        let finalized = finalize(component);

        assert_eq!(collect_expr_ids(&finalized), vec![0, 1, 2]);
    }

    #[test]
    fn renumber_expr_ids_is_deterministic_in_place() {
        let mut component = expr(make_expr(3, Op::Add, &["i"], "i"))
            .fanout(expr(make_expr(8, Op::Mul, &["i", "i"], "i")))
            .pair(expr(make_expr(13, Op::Sub, &["i"], "i")));

        renumber_expr_ids(&mut component);
        assert_eq!(collect_expr_ids(&component), vec![0, 1, 2]);

        renumber_expr_ids(&mut component);
        assert_eq!(collect_expr_ids(&component), vec![0, 1, 2]);
    }

    fn collect_expr_ids(component: &Component) -> Vec<usize> {
        let mut ids = Vec::new();
        collect_expr_ids_inner(component, &mut ids);
        ids
    }

    fn collect_expr_ids_inner(component: &Component, ids: &mut Vec<usize>) {
        match component {
            Component::Expr(expr) => ids.push(expr.id),
            Component::Compose(left, right)
            | Component::Chain(left, right)
            | Component::Fanout(left, right)
            | Component::Pair(left, right) => {
                collect_expr_ids_inner(left, ids);
                collect_expr_ids_inner(right, ids);
            }
            Component::Swap(inner) => collect_expr_ids_inner(inner, ids),
        }
    }

    fn make_expr(id: usize, op: Op, inputs: &[&str], output: &str) -> Expr {
        Expr {
            id,
            op,
            inputs: inputs
                .iter()
                .map(|pattern| Pattern(pattern.chars().collect()))
                .collect(),
            output: Pattern(output.chars().collect()),
            schedule: Schedule::default(),
        }
    }
}

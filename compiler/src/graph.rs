use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::ast::{
    BinaryOp, Combinator, Expr, ExprBank, ExprRef, IndexExpr, NoOp, ScalarOp, Schedule, UnaryOp,
};

static NODE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

type NodeRef = Arc<Mutex<Node>>;

#[derive(Clone, Debug)]
pub enum NodeBody {
    Leaf,
    Interior {
        op: char,
        schedule: Schedule,
        shape: Vec<(usize, usize)>,
    },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub index: String,
    pub body: NodeBody,
    parents: Vec<NodeRef>,
    children: Vec<(NodeRef, String)>,
}

impl Node {
    pub fn children(&self) -> Vec<(Node, String)> {
        self.children
            .iter()
            .map(|(child_ref, index)| (child_ref.lock().unwrap().clone(), index.clone()))
            .collect()
    }
}

fn get_parent_of_leftmost_leaf(node: &NodeRef) -> Option<NodeRef> {
    let mut current = Arc::clone(node);
    let mut parent = None;

    loop {
        let next = {
            let node = current.lock().unwrap();
            if node.children.is_empty() {
                return parent;
            }
            Arc::clone(&node.children[0].0)
        };
        parent = Some(current);
        current = next;
    }
}

fn get_leftmost_leaf(node_ref: &NodeRef) -> NodeRef {
    let mut current = Arc::clone(node_ref);
    loop {
        let next = {
            let node = current.lock().unwrap();
            if node.children.is_empty() {
                return Arc::clone(&current);
            }
            Arc::clone(&node.children[0].0)
        };
        current = next;
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    roots: Vec<NodeRef>,
}

impl Graph {
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    pub fn deepcopy(&self) -> Self {
        fn copy_recursive(
            node_ref: &NodeRef,
            visited: &mut HashMap<*const Node, NodeRef>,
        ) -> NodeRef {
            let node = node_ref.lock().unwrap();
            let ptr = &*node as *const Node;
            if let Some(n) = visited.get(&ptr) {
                return Arc::clone(n);
            }

            let new_node = Arc::new(Mutex::new(Node {
                id: node.id,
                index: node.index.clone(),
                body: node.body.clone(),
                parents: Vec::new(),
                children: Vec::new(),
            }));
            visited.insert(ptr, Arc::clone(&new_node));

            let children: Vec<_> = node
                .children
                .iter()
                .map(|(c, idx)| (copy_recursive(c, visited), idx.clone()))
                .collect();
            let parents: Vec<_> = node
                .parents
                .iter()
                .map(|p| copy_recursive(p, visited))
                .collect();

            let mut new_lock = new_node.lock().unwrap();
            new_lock.children = children;
            new_lock.parents = parents;
            drop(new_lock);

            new_node
        }
        let mut visited = HashMap::new();
        let roots = self
            .roots
            .iter()
            .map(|r| copy_recursive(r, &mut visited))
            .collect();
        Self { roots }
    }

    pub fn roots(&self) -> Vec<NodeRef> {
        self.roots.iter().cloned().collect()
    }

    pub fn root(&self) -> NodeRef {
        Arc::clone(self.roots.last().expect("Graph has no roots"))
    }

    pub fn from_expr_bank(expr_bank: &ExprBank) -> Graph {
        let mut graph = Self::new();
        let root =
            graph.from_expr_ref_with_expr_bank(&ExprRef(expr_bank.0.len() - 1), expr_bank, vec![]);
        graph.roots.push(root);
        graph
    }

    pub fn chain(&self, other: &Self) -> Self {
        other.compose(self)
    }

    pub fn compose(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        let mut right = other.deepcopy();

        let mut r_iter = right.roots.into_iter();
        let map: HashMap<usize, NodeRef> = left
            .leaves()
            .into_iter()
            .filter_map(|leaf| {
                r_iter
                    .next()
                    .map(|root| (Arc::as_ptr(&leaf) as usize, root))
            })
            .collect();

        let mut seen = HashSet::new();
        let mut stack = left.roots.clone();
        while let Some(node) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&node) as usize) {
                continue;
            }
            let mut n = node.lock().unwrap();
            for (child, _) in &mut n.children {
                if let Some(repl) = map.get(&(Arc::as_ptr(child) as usize)) {
                    *child = repl.clone();
                }
            }
            stack.extend(n.children.iter().map(|(c, _)| c.clone()));
        }

        left.roots.extend(r_iter);
        left
    }

    pub fn fanout(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        let mut right = other.deepcopy();

        let map: HashMap<usize, NodeRef> = right
            .leaves()
            .into_iter()
            .zip(left.leaves())
            .map(|(r, l)| (Arc::as_ptr(&r) as usize, l))
            .collect();

        let mut seen = HashSet::new();
        let mut stack = right.roots.clone();
        while let Some(node) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&node) as usize) {
                continue;
            }
            let mut n = node.lock().unwrap();
            for (child, _) in &mut n.children {
                if let Some(repl) = map.get(&(Arc::as_ptr(child) as usize)) {
                    *child = repl.clone();
                }
            }
            stack.extend(n.children.iter().map(|(c, _)| c.clone()));
        }

        left.roots.extend(right.roots);
        left
    }

    pub fn pair(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        left.roots.extend(other.deepcopy().roots);
        left
    }

    fn add_node(
        &mut self,
        index: String,
        body: NodeBody,
        parents: Vec<NodeRef>,
        children: Vec<(NodeRef, String)>,
    ) -> NodeRef {
        let node = Arc::new(Mutex::new(Node {
            id: NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            index: index.clone(),
            body,
            parents: parents.clone(),
            children,
        }));

        for p in parents {
            p.lock()
                .unwrap()
                .children
                .push((Arc::clone(&node), index.clone()));
        }

        node
    }

    fn from_expr_ref_with_expr_bank(
        &mut self,
        expr_ref: &ExprRef,
        expr_bank: &ExprBank,
        parents: Vec<NodeRef>,
    ) -> NodeRef {
        let Some(expr) = &expr_bank.0.get(expr_ref.0) else {
            panic!("Expression Bank is empty.")
        };

        match expr {
            Expr::Index(IndexExpr { op, out, schedule }) => {
                let children = match op {
                    ScalarOp::BinaryOp(BinaryOp::Add(in0, in1))
                    | ScalarOp::BinaryOp(BinaryOp::Sub(in0, in1))
                    | ScalarOp::BinaryOp(BinaryOp::Mul(in0, in1))
                    | ScalarOp::BinaryOp(BinaryOp::Div(in0, in1))
                    | ScalarOp::BinaryOp(BinaryOp::Max(in0, in1))
                    | ScalarOp::BinaryOp(BinaryOp::Min(in0, in1)) => vec![
                        (
                            self.add_node(in0.0.clone(), NodeBody::Leaf, vec![], vec![]),
                            in0.0.clone(),
                        ),
                        (
                            self.add_node(in1.0.clone(), NodeBody::Leaf, vec![], vec![]),
                            in1.0.clone(),
                        ),
                    ],
                    ScalarOp::UnaryOp(UnaryOp::Accum(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Prod(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Relu(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Neg(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Max(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Min(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Recip(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Exp(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Log(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Sqrt(in0))
                    | ScalarOp::UnaryOp(UnaryOp::Abs(in0))
                    | ScalarOp::NoOp(NoOp(in0)) => {
                        vec![(
                            self.add_node(in0.0.clone(), NodeBody::Leaf, vec![], vec![]),
                            in0.0.clone(),
                        )]
                    }
                };
                let op = match op {
                    ScalarOp::UnaryOp(UnaryOp::Accum(_))
                    | ScalarOp::BinaryOp(BinaryOp::Add(_, _)) => '+',
                    ScalarOp::UnaryOp(UnaryOp::Prod(_))
                    | ScalarOp::BinaryOp(BinaryOp::Mul(_, _)) => '*',
                    ScalarOp::UnaryOp(UnaryOp::Relu(_)) => '!',
                    ScalarOp::UnaryOp(UnaryOp::Max(_))
                    | ScalarOp::BinaryOp(BinaryOp::Max(_, _)) => '>',
                    ScalarOp::UnaryOp(UnaryOp::Min(_))
                    | ScalarOp::BinaryOp(BinaryOp::Min(_, _)) => '<',
                    ScalarOp::UnaryOp(UnaryOp::Neg(_))
                    | ScalarOp::BinaryOp(BinaryOp::Sub(_, _)) => '-',
                    ScalarOp::UnaryOp(UnaryOp::Recip(_))
                    | ScalarOp::BinaryOp(BinaryOp::Div(_, _)) => '/',
                    ScalarOp::UnaryOp(UnaryOp::Exp(_)) => '^',
                    ScalarOp::UnaryOp(UnaryOp::Log(_)) => '$',
                    ScalarOp::UnaryOp(UnaryOp::Sqrt(_)) => '@',
                    ScalarOp::UnaryOp(UnaryOp::Abs(_)) => '#',
                    ScalarOp::NoOp(_) => ' ',
                };

                let body = NodeBody::Interior {
                    op,
                    schedule: schedule.clone(),
                    shape: infer_shape(&out.0, children.iter().map(|child| &child.1).collect()),
                };
                self.add_node(out.0.clone(), body, parents, children)
            }
            Expr::Combinator(Combinator::Chain(left_ref, right_ref)) => {
                let left = self.from_expr_ref_with_expr_bank(left_ref, expr_bank, parents.clone());
                let right = self.from_expr_ref_with_expr_bank(right_ref, expr_bank, parents);

                if let Some(parent) = get_parent_of_leftmost_leaf(&right) {
                    let mut pn = parent.lock().unwrap();
                    let (orphan, tag) = pn.children[0].clone();
                    pn.children[0] = (Arc::clone(&left), tag);
                    drop(pn);
                    left.lock().unwrap().parents.push(Arc::clone(&parent));
                    drop(orphan);
                }

                right
            }
        }
    }

    fn leaves(&self) -> Vec<NodeRef> {
        use std::collections::HashSet;
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = self.roots.clone();

        while let Some(n) = stack.pop() {
            let ptr = {
                let node = n.lock().unwrap();
                &*node as *const _ as usize
            };
            if !seen.insert(ptr) {
                continue;
            }
            let cs = {
                let node = n.lock().unwrap();
                if node.children.is_empty() {
                    out.push(n.clone());
                    Vec::new()
                } else {
                    node.children.iter().rev().map(|(c, _)| c.clone()).collect()
                }
            };
            stack.extend(cs);
        }
        out
    }

    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph G {\n");
        let mut visited = HashSet::new();
        for root in &self.roots {
            Self::dot_node(&root.lock().unwrap(), &mut visited, &mut out);
        }
        out.push('}');
        out
    }

    fn dot_node(node: &Node, visited: &mut HashSet<usize>, out: &mut String) {
        let id = node as *const _ as usize;
        if !visited.insert(id) {
            return;
        }
        let label = match &node.body {
            NodeBody::Leaf => format!("{}", node.index),
            NodeBody::Interior { op, .. } => format!("{} {}", node.index, op),
        };
        writeln!(out, "\t{} [label=\"{}\"];", id, label).unwrap();
        for (child, edge_label) in &node.children {
            let child_node = child.lock().unwrap();
            let child_id = &*child_node as *const _ as usize;
            writeln!(out, "\t{} -> {} [label=\"{}\"];", id, child_id, edge_label).unwrap();
            Self::dot_node(&child_node, visited, out);
        }
    }
}

fn infer_shape(index: &String, child_indices: Vec<&String>) -> Vec<(usize, usize)> {
    let index_map: HashMap<char, (usize, usize)> = child_indices
        .iter()
        .enumerate()
        .rev()
        .flat_map(|(child_ind, child_index)| {
            child_index
                .chars()
                .rev()
                .enumerate()
                .map(move |(char_ind, c)| (c, (child_ind, child_index.len() - 1 - char_ind)))
        })
        .collect();

    index.chars().map(|c| index_map[&c]).collect()
}

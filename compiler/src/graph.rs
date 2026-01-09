use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::ast::{BinaryOp, Expr, ExprBank, ExprRef, NoOp, ScalarOp, UnaryOp};

static NODE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bound {
    Base,
    Factor(usize),
}

/// Used to associate objects to a particular domain `(input_ind, dim_ind)`.
/// Depending on context, can be used as a local address (where `input_ind`
/// refers to the input index of a Node's child list or in as a global address
/// where `input_ind` refers to the index of a Graph's leaves list.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShapeAddr {
    pub input_ind: usize,
    pub dim_ind: usize,
}

#[derive(Clone, Debug)]
pub struct LoopSpec {
    pub group: usize, // base loops and their corresponding split loops share a group
    pub ind: usize,   // index within group, 0 reserved for base loop, splits ordered by declaration
    pub output_dim: Option<usize>, // None only for reduction dimensions
    pub addrs: Vec<ShapeAddr>, // (input index, dimension index)
    pub split_factors: Vec<usize>,
    pub bound: Bound,
    pub index_reconstruction: Option<Vec<usize>>, // contains split factors necessary to reconstruct
}

#[derive(Clone, Debug)]
pub enum NodeBody {
    Leaf,
    Interior {
        op: char,
        // outer vec runs over output dims, inner vec runs over possible origin
        // domains in order of appearance in the input list, e.g., in
        // `ik*kj~ijk`, the output index `k` has two possible origin domains:
        // `ShapeAddr{input_ind: 0, dim_ind: 1}` or
        // `ShapeAddr{input_ind: 1, dim_ind: 0}`
        shape_addr_lists: Vec<Vec<ShapeAddr>>,
        split_factor_lists: Vec<Vec<usize>>,
        loop_specs: Vec<LoopSpec>,
        compute_levels: Vec<usize>, // compute-levels of children (0 reserved for non-fused)
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

type NodeRef = Arc<Mutex<Node>>;

impl Node {
    pub fn children(&self) -> Vec<(Node, String)> {
        self.children
            .iter()
            .map(|(child_ref, index)| (child_ref.lock().unwrap().clone(), index.clone()))
            .collect()
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
        let right = other.deepcopy();

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
        let right = other.deepcopy();

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

        let Expr { op, out, schedule } = expr;
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
            ScalarOp::UnaryOp(UnaryOp::Accum(_)) | ScalarOp::BinaryOp(BinaryOp::Add(_, _)) => '+',
            ScalarOp::UnaryOp(UnaryOp::Prod(_)) | ScalarOp::BinaryOp(BinaryOp::Mul(_, _)) => '*',
            ScalarOp::UnaryOp(UnaryOp::Relu(_)) => '!',
            ScalarOp::UnaryOp(UnaryOp::Max(_)) | ScalarOp::BinaryOp(BinaryOp::Max(_, _)) => '>',
            ScalarOp::UnaryOp(UnaryOp::Min(_)) | ScalarOp::BinaryOp(BinaryOp::Min(_, _)) => '<',
            ScalarOp::UnaryOp(UnaryOp::Neg(_)) | ScalarOp::BinaryOp(BinaryOp::Sub(_, _)) => '-',
            ScalarOp::UnaryOp(UnaryOp::Recip(_)) | ScalarOp::BinaryOp(BinaryOp::Div(_, _)) => '/',
            ScalarOp::UnaryOp(UnaryOp::Exp(_)) => '^',
            ScalarOp::UnaryOp(UnaryOp::Log(_)) => '$',
            ScalarOp::UnaryOp(UnaryOp::Sqrt(_)) => '@',
            ScalarOp::UnaryOp(UnaryOp::Abs(_)) => '#',
            ScalarOp::NoOp(_) => ' ',
        };

        let mut shape_table: HashMap<char, (Vec<ShapeAddr>, Vec<usize>)> = HashMap::new();
        for (input_ind, (_, child_index)) in children.iter().enumerate() {
            for (dim_ind, c) in child_index.chars().enumerate() {
                let entry = shape_table.entry(c).or_insert_with(|| {
                    (
                        Vec::new(),
                        schedule.splits.get(&c).cloned().unwrap_or_else(|| vec![]),
                    )
                });

                entry.0.push(ShapeAddr { input_ind, dim_ind });
            }
        }

        let (shape_addr_lists, split_factor_lists): (Vec<Vec<ShapeAddr>>, Vec<Vec<usize>>) =
            out.0.chars().map(|c| shape_table[&c].clone()).unzip();

        let char_index_to_output_dim: HashMap<char, usize> =
            out.0.chars().enumerate().map(|(i, c)| (c, i)).collect();

        let mut unique_char_indices = HashSet::<char>::new();
        let char_indexes: Vec<char> = out
            .0
            .chars()
            .chain(
                children
                    .iter()
                    .flat_map(|(_, child_index)| child_index.chars()),
            )
            .filter(move |c| unique_char_indices.insert(*c))
            .collect();

        let loop_groups: HashMap<char, usize> = char_indexes
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i))
            .collect();

        // TODO this could be cleaned up by computing a loop order first and then unifying
        //      these branches
        let loop_specs: Vec<LoopSpec> = if schedule.loop_order.is_empty() {
            char_indexes
                .iter()
                .flat_map(|c| {
                    let loop_group = loop_groups[&c];
                    let output_dim = char_index_to_output_dim.get(&c).copied();
                    let (shape_addrs, split_factors) = &shape_table[&c];
                    let base_loop_spec = LoopSpec {
                        group: loop_group,
                        ind: 0,
                        output_dim,
                        addrs: shape_addrs.clone(),
                        split_factors: split_factors.clone(),
                        bound: Bound::Base,
                        index_reconstruction: None, // TODO
                    };

                    let mut enumerated_split_factors = split_factors.iter().enumerate();

                    let index_reconstructed_loop_spec =
                        enumerated_split_factors
                            .next()
                            .map(move |(ind, _factor)| LoopSpec {
                                group: loop_group,
                                ind,
                                output_dim,
                                addrs: shape_addrs.clone(),
                                split_factors: split_factors.clone(),
                                bound: Bound::Factor(ind),
                                index_reconstruction: if split_factors.is_empty() {
                                    None
                                } else {
                                    Some(split_factors.clone())
                                },
                            });

                    let remaining_factor_loop_specs =
                        enumerated_split_factors.map(move |(ind, _factor)| LoopSpec {
                            group: loop_group,
                            ind,
                            output_dim,
                            addrs: shape_addrs.clone(),
                            split_factors: split_factors.clone(),
                            bound: Bound::Factor(ind),
                            index_reconstruction: None,
                        });

                    std::iter::once(base_loop_spec)
                        .chain(index_reconstructed_loop_spec.into_iter())
                        .chain(remaining_factor_loop_specs)
                })
                .collect()
        } else {
            let mut index_reconstructed_groups = HashSet::<usize>::new();
            let mut loop_specs: Vec<LoopSpec> = schedule
                .loop_order
                .iter()
                .rev()
                .map(|(c, split_factor_ind)| {
                    let loop_group = loop_groups[&c];
                    let output_dim = char_index_to_output_dim.get(&c).copied();
                    let (shape_addrs, split_factors) = &shape_table[&c];

                    let bound = match (split_factors.is_empty(), split_factor_ind) {
                        (false, 0) | (true, _) => Bound::Base,
                        (false, ind) => Bound::Factor(*ind),
                    };

                    let index_reconstruction = if !split_factors.is_empty()
                        && index_reconstructed_groups.insert(loop_group)
                    {
                        Some(split_factors.clone())
                    } else {
                        None
                    };

                    LoopSpec {
                        group: loop_group,
                        ind: *split_factor_ind,
                        output_dim,
                        addrs: shape_addrs.clone(),
                        split_factors: split_factors.clone(),
                        bound,
                        index_reconstruction,
                    }
                })
                .collect();
            loop_specs.reverse();
            loop_specs
        };

        let mut compute_levels = schedule.compute_levels.clone();
        compute_levels.resize(children.len(), 0);

        //// TODO validate number of loop_specs: one per unique char index plus one per split

        let body = NodeBody::Interior {
            op,
            shape_addr_lists,
            split_factor_lists,
            loop_specs,
            compute_levels,
        };
        self.add_node(out.0.clone(), body, parents, children)
    }

    pub fn leaves(&self) -> Vec<NodeRef> {
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

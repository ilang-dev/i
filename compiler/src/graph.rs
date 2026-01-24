use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::ast::{Expr, Op};

static NODE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Bound {
    Base,
    Split {
        factor: usize,
        ind: usize, // 1-index (0 reserved for base loop)
    },
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

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Axis {
    pub addrs: Vec<ShapeAddr>,
    pub kind: Bound,
}

#[derive(Clone, Debug)]
pub struct LoopSpec {
    pub output_dim: Option<usize>, // None only for reduction dimensions
    pub addrs: Vec<ShapeAddr>,     // (input index, dimension index)
    pub split_factors: Vec<usize>,
    pub bound: Bound,
    pub axis: Axis,
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
        logical_shape: Vec<ShapeAddr>,
        physical_shape: Vec<Axis>,
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
    pub roots: Vec<NodeRef>,
    pub inputs: Vec<NodeRef>,
}

impl Graph {
    pub fn deepcopy(&self) -> Self {
        fn copy_recursive(
            node_ref: &NodeRef,
            visited: &mut HashMap<*const Node, NodeRef>,
            input_map: &HashMap<usize, usize>,
            inputs: &mut Vec<Option<NodeRef>>,
        ) -> NodeRef {
            let node = node_ref.lock().unwrap();
            let ptr = &*node as *const Node;
            if let Some(n) = visited.get(&ptr) {
                return Arc::clone(n);
            }

            let new_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
            let new_node = Arc::new(Mutex::new(Node {
                id: new_id,
                index: node.index.clone(),
                body: node.body.clone(),
                children: Vec::new(),
            }));
            visited.insert(ptr, Arc::clone(&new_node));

            if let Some(&ind) = input_map.get(&node.id) {
                inputs[ind] = Some(new_node.clone());
            }

            let children: Vec<_> = node
                .children
                .iter()
                .map(|(c, idx)| (copy_recursive(c, visited, input_map, inputs), idx.clone()))
                .collect();

            let mut new_lock = new_node.lock().unwrap();
            new_lock.children = children;
            drop(new_lock);

            new_node
        }

        let input_map: HashMap<usize, usize> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(ind, node)| (node.lock().unwrap().id, ind))
            .collect();

        let mut inputs: Vec<Option<NodeRef>> = vec![None; self.inputs.len()];

        let mut visited = HashMap::new();
        let roots = self
            .roots
            .iter()
            .map(|r| copy_recursive(r, &mut visited, &input_map, &mut inputs))
            .collect();

        let inputs: Vec<NodeRef> = inputs.into_iter().map(Option::unwrap).collect();

        Self { roots, inputs }
    }

    pub fn roots(&self) -> Vec<NodeRef> {
        self.roots.iter().cloned().collect()
    }

    pub fn chain(&self, other: &Self) -> Self {
        other.compose(self)
    }

    pub fn compose(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        let right = other.deepcopy();

        let mut right_roots = right.roots.into_iter();
        let left_inputs = left.inputs.into_iter();
        let mut inputs = right.inputs;

        for input in left_inputs {
            if let Some(root) = right_roots.next() {
                let root_snapshot = { root.lock().unwrap().clone() };
                *input.lock().unwrap() = root_snapshot;
            } else {
                inputs.push(input)
            }
        }

        left.roots.extend(right_roots);
        left.inputs = inputs;
        left
    }

    pub fn fanout(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        let right = other.deepcopy();

        let mut left_inputs = left.inputs.into_iter();
        let mut right_inputs = right.inputs.into_iter();
        let mut inputs = Vec::new();

        for input in left_inputs {
            if let Some(other_input) = right_inputs.next() {
                let snapshot = { input.lock().unwrap().clone() };
                *other_input.lock().unwrap() = snapshot;
                inputs.push(input);
            } else {
                inputs.push(input);
            }
        }

        inputs.extend(right_inputs);
        left.roots.extend(right.roots);
        left.inputs = inputs;
        left
    }

    pub fn pair(&self, other: &Self) -> Self {
        let mut left = self.deepcopy();
        let right = other.deepcopy();
        left.roots.extend(right.roots);
        left.inputs.extend(right.inputs);
        left
    }

    fn add_node(
        &mut self,
        index: String,
        body: NodeBody,
        children: Vec<(NodeRef, String)>,
    ) -> NodeRef {
        Arc::new(Mutex::new(Node {
            id: NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            index: index.clone(),
            body,
            children,
        }))
    }

    pub fn from_expr(expr: &Expr) -> Graph {
        let mut graph = Self {
            roots: Vec::new(),
            inputs: Vec::new(),
        };

        let Expr { op, out, schedule } = expr;
        let children: Vec<(NodeRef, String)> = op
            .args
            .iter()
            .map(|s| {
                (
                    graph.add_node(s.0.clone(), NodeBody::Leaf, vec![]),
                    s.0.clone(),
                )
            })
            .collect();

        let op = match op.op {
            Op::NoOp => ' ',
            Op::Add => '+',
            Op::Sub => '-',
            Op::Mul => '*',
            Op::Div => '/',
            Op::Max => '>',
            Op::Min => '<',
            Op::Relu => '!',
            Op::Exp => '^',
            Op::Log => '$',
            Op::Sqrt => '@',
            Op::Abs => '#',
            Op::Neg => '-',
            Op::Recip => '/',
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
                    let output_dim = char_index_to_output_dim.get(&c).copied();
                    let (shape_addrs, split_factors) = &shape_table[&c];
                    let base_loop_spec = LoopSpec {
                        output_dim,
                        addrs: shape_addrs.clone(),
                        split_factors: split_factors.clone(),
                        bound: Bound::Base,
                        axis: Axis {
                            addrs: shape_addrs.clone(),
                            kind: match split_factors.is_empty() {
                                true => Bound::Base,
                                false => Bound::Split { factor: 0, ind: 0 },
                            },
                        },
                        index_reconstruction: None, // TODO
                    };

                    let mut enumerated_split_factors = split_factors.iter().enumerate();

                    let index_reconstructed_loop_spec =
                        enumerated_split_factors
                            .next()
                            .map(move |(ind, &factor)| LoopSpec {
                                output_dim,
                                addrs: shape_addrs.clone(),
                                split_factors: split_factors.clone(),
                                bound: Bound::Split { factor, ind },
                                axis: Axis {
                                    addrs: shape_addrs.clone(),
                                    kind: Bound::Split { factor, ind },
                                },
                                index_reconstruction: if split_factors.is_empty() {
                                    None
                                } else {
                                    Some(split_factors.clone())
                                },
                            });

                    let remaining_factor_loop_specs =
                        enumerated_split_factors.map(move |(ind, &factor)| LoopSpec {
                            output_dim,
                            addrs: shape_addrs.clone(),
                            split_factors: split_factors.clone(),
                            bound: Bound::Split { factor, ind },
                            axis: Axis {
                                addrs: shape_addrs.clone(),
                                kind: Bound::Split { factor, ind },
                            },
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

                    let bound = match split_factors.is_empty() {
                        true => Bound::Base,
                        false => Bound::Split {
                            factor: match split_factor_ind {
                                0 => 0,
                                _ => split_factors[split_factor_ind - 1],
                            },
                            ind: *split_factor_ind,
                        },
                    };

                    let index_reconstruction = if !split_factors.is_empty()
                        && index_reconstructed_groups.insert(loop_group)
                    {
                        Some(split_factors.clone())
                    } else {
                        None
                    };

                    LoopSpec {
                        output_dim,
                        addrs: shape_addrs.clone(),
                        split_factors: split_factors.clone(),
                        bound,
                        axis: Axis {
                            addrs: shape_addrs.clone(),
                            kind: bound,
                        },
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

        let logical_shape: Vec<ShapeAddr> = shape_addr_lists
            .iter()
            .map(|list| list[0]) // prefer earliest instance
            .collect();

        let physical_shape: Vec<Axis> = shape_addr_lists
            .iter()
            .zip(split_factor_lists.iter())
            .flat_map(|(addrs, split_factors)| {
                std::iter::once((0usize, 0usize))
                    .chain(
                        split_factors
                            .iter()
                            .enumerate()
                            .map(|(ind, &factor)| (ind + 1, factor)),
                    )
                    .map(move |(ind, factor)| Axis {
                        addrs: addrs.clone(),
                        kind: if split_factors.is_empty() {
                            Bound::Base
                        } else {
                            Bound::Split { factor, ind }
                        },
                    })
            })
            .collect();

        let body = NodeBody::Interior {
            op,
            shape_addr_lists,
            logical_shape,
            physical_shape,
            split_factor_lists,
            loop_specs,
            compute_levels,
        };
        let root = graph.add_node(out.0.clone(), body, children);

        graph.roots.push(root);
        graph.inputs.extend(graph.leaves());
        graph
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

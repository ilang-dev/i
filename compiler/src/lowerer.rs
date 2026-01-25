use std::collections::{HashMap, HashSet};

use crate::block::{Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Axis, Bound, Graph, LoopSpec, Node, NodeBody, ShapeAddr};

#[derive(Clone, Debug)]
struct Arg {
    type_: ArgType,
    physical_shape: Vec<Axis>,
    ident: Expr,
}

#[derive(Clone, Debug)]
enum ArgType {
    ReadOnly,
    Writeable,
}

fn get_addr_pref_table(graph: &Graph) -> HashMap<ShapeAddr, ShapeAddr> {
    let mut node_to_leaf_ind: HashMap<usize, usize> = graph
        .inputs
        .iter()
        .enumerate()
        .map(|(ind, node)| (node.lock().unwrap().id, ind))
        .collect();

    let mut table = HashMap::new();

    // updates table, returns semantic shape of node
    fn _f(
        node: &Node,
        table: &mut HashMap<ShapeAddr, ShapeAddr>,
        node_to_leaf_ind: &HashMap<usize, usize>,
    ) -> Vec<ShapeAddr> {
        match &node.body {
            NodeBody::Leaf => {
                let leaf_ind = node_to_leaf_ind[&node.id];
                (0..node.rank)
                    .map(|dim_ind| {
                        let addr = ShapeAddr {
                            input_ind: leaf_ind,
                            dim_ind: dim_ind,
                        };
                        table.insert(addr, addr);
                        addr
                    })
                    .collect()
            }
            NodeBody::Interior {
                shape_addr_lists, ..
            } => {
                let child_shapes: Vec<Vec<ShapeAddr>> = node
                    .childs()
                    .iter()
                    .map(|node| _f(&node.lock().unwrap(), table, node_to_leaf_ind))
                    .collect();
                shape_addr_lists
                    .iter()
                    .map(|list| {
                        // update pref table, return preferred
                        assert!(list.len() > 0);
                        let list: Vec<ShapeAddr> = list
                            .iter()
                            .map(|addr| child_shapes[addr.input_ind][addr.dim_ind])
                            .collect(); // globalize list
                        let pref = list[0];
                        let pref_pairs = (1..list.len()).map(|i| (list[i], pref));
                        table.extend(pref_pairs);
                        pref
                    })
                    .collect()
            }
        }
    }

    for root in graph.roots() {
        _f(&root.lock().unwrap(), &mut table, &node_to_leaf_ind);
    }

    table
}

// TODO write a real docstring here
// This function is responsible for the rank, shape, and exec functions.
// `rank` and `shape` are easy, but `exec` has some complexity. The API should
// be `void exec(const Tensor* inputs, size_t n_inputs, TensorMut* output)`. It
// must map any relevant input values to idents (tensors and dims), perform any
// allocations (and eventually frees), and launch kernels
pub fn lower(graph: &Graph) -> Program {
    let mut topo_ind = 0;
    let mut library = Block::default();
    let mut exec_block = Block::default();
    let mut node_to_leaf_ind: HashMap<usize, usize> = graph
        .inputs
        .iter()
        .enumerate()
        .map(|(ind, node)| (node.lock().unwrap().id, ind))
        .collect();
    //let mut shape_addr_preference = HashMap::<ShapeAddr, ShapeAddr>::new();
    let shape_addr_preference = get_addr_pref_table(graph);

    let lowereds: Vec<(Vec<ShapeAddr>, Vec<Axis>, Expr, Block)> = graph
        .roots()
        .iter()
        .enumerate()
        .map(|(root_ind, node)| {
            lower_node(
                &node.lock().unwrap(),
                Some(root_ind),
                &mut topo_ind,
                &mut library,
                &mut exec_block,
                &mut node_to_leaf_ind,
                &shape_addr_preference,
                HashSet::new(),
                &mut vec![],
            )
        })
        .collect();

    let shape_exprs: Vec<Vec<Expr>> = lowereds
        .iter()
        .map(|lowered| lowered.0.clone())
        .map(|shape_addr_list| shape_addr_list.iter().map(input_shape_expr).collect())
        .collect();

    Program {
        count: count(graph.roots().len()),
        //ranks: ranks(lowereds.iter().map(|l| l.0).collect()),
        ranks: ranks(
            graph
                .roots()
                .iter()
                .map(|root| root.lock().unwrap().rank)
                .collect(),
        ),
        shapes: shapes(shape_exprs),
        library: library,
        exec: Statement::Function {
            signature: FunctionSignature::Exec,
            body: exec_block,
        },
    }
}

/// Lower node. Update library and exec block, return (semantic shape, physical shape, buffer ident, fused fragment)
fn lower_node(
    node: &Node,
    root_ind: Option<usize>,
    topo_ind: &mut usize,
    library: &mut Block,
    exec_block: &mut Block,
    node_to_leaf_ind: &mut HashMap<usize, usize>,
    shape_addr_preference: &HashMap<ShapeAddr, ShapeAddr>,
    prunable_axes: HashSet<Axis>,
    args: &mut Vec<(Arg, usize)>,
) -> (Vec<ShapeAddr>, Vec<Axis>, Expr, Block) {
    let NodeBody::Interior {
        op,
        shape_addr_lists,
        logical_shape,
        physical_shape,
        split_factor_lists,
        loop_specs,
        compute_levels,
    } = &node.body
    else {
        assert!(prunable_axes.is_empty(), "Cannot fuse leaf nodes.");
        // handle leaf nodes
        let leaf_ind = node_to_leaf_ind[&node.id];
        let logical_shape: Vec<ShapeAddr> = (0..node.rank)
            .map(|dim_ind| ShapeAddr {
                input_ind: leaf_ind,
                dim_ind: dim_ind,
            })
            .collect();

        let physical_shape: Vec<Axis> = logical_shape
            .iter()
            .map(|addr| Axis {
                addrs: vec![*addr],
                kind: Bound::Base,
            })
            .collect();

        let buffer_ident = Expr::Indexed {
            expr: Box::new(Expr::Ident("inputs".into())),
            index: Box::new(Expr::Int(leaf_ind)),
        };

        return (
            logical_shape,
            physical_shape,
            buffer_ident,
            Block::default(),
        );
    };

    assert!(
        node.children().len() == compute_levels.len(),
        "Number of compute level specifications does not match number of children"
    );

    // non-fused -> start new arg lists, fused -> use existing list
    let args = if prunable_axes.is_empty() {
        &mut vec![]
    } else {
        args
    };

    let children_lowereds: Vec<(Vec<ShapeAddr>, Vec<Axis>, Expr, Block)> = node
        .children()
        .iter()
        .zip(compute_levels.iter())
        .enumerate()
        .map(|(child_ind, ((node, _), &compute_level))| {
            lower_node(
                &node,
                None,
                topo_ind,
                library,
                exec_block,
                node_to_leaf_ind,
                shape_addr_preference,
                get_prunable_axes(&loop_specs, compute_level, child_ind),
                args,
            )
        })
        .collect();

    let child_shape_addr_lists: Vec<Vec<ShapeAddr>> =
        children_lowereds.iter().map(|l| l.0.clone()).collect();
    let child_physical_shapes: Vec<Vec<Axis>> =
        children_lowereds.iter().map(|l| l.1.clone()).collect();
    let child_buffer_idents: Vec<Expr> = children_lowereds.iter().map(|l| l.2.clone()).collect();
    let child_fragments: Vec<Block> = children_lowereds.iter().map(|l| l.3.clone()).collect();

    let library_function_ident = Expr::Ident(format!("f{topo_ind}"));
    let buffer_ident = match root_ind {
        None => Expr::Ident(format!("s{topo_ind}")),
        Some(root_ind) => Expr::Indexed {
            expr: Box::new(Expr::Ident("outputs".into())),
            index: Box::new(Expr::Int(root_ind)),
        },
    };

    // semantic shape addrs of current node without regard to splitting or fusion
    let shape_addrs: Vec<ShapeAddr> = logical_shape
        .iter()
        .map(|ShapeAddr { input_ind, dim_ind }| child_shape_addr_lists[*input_ind][*dim_ind])
        .collect();

    let prunable_axes: HashSet<Axis> = prunable_axes
        .iter()
        .map(|Axis { addrs, kind }| Axis {
            addrs: addrs
                .iter()
                .flat_map(|addr| shape_addr_lists[addr.dim_ind].clone())
                .collect(),
            kind: *kind,
        })
        .collect();

    let physical_shape: Vec<Axis> = match root_ind.is_some() {
        // roots always have semantic shape
        true => shape_addr_lists
            .iter()
            .map(|addrs| Axis {
                addrs: addrs.clone(),
                kind: Bound::Base,
            })
            .collect(),
        // other interior node can be folded
        false => physical_shape
            .iter()
            .filter(|axis| !prunable_axes.contains(&globalize_axis(axis, &child_shape_addr_lists)))
            .cloned()
            .collect(),
    };

    // mutable for updating for child indexing exprs
    let mut addr_to_split_factor_list: HashMap<&ShapeAddr, &Vec<usize>> =
        shape_addrs.iter().zip(split_factor_lists.iter()).collect();

    // create allocation
    if root_ind.is_none() {
        exec_block.statements.push(Statement::Alloc {
            index: *topo_ind,
            initial_value: Box::new(Expr::Scalar(match op {
                '>' => f32::NEG_INFINITY,
                '<' => f32::INFINITY,
                '*' => 1.,
                _ => 0.,
            })),
            shape: build_buffer_shape_exprs(
                &physical_shape
                    .iter()
                    .map(|axis| globalize_axis(axis, &child_shape_addr_lists))
                    .collect(),
                &split_factor_lists,
                &child_shape_addr_lists,
                &addr_to_split_factor_list,
            ),
        });
    }

    let (mut readonly_offset, mut writeable_offset) =
        args.iter()
            .map(|(arg, _)| arg)
            .fold(
                (0, 0),
                |(readonly_offset, writeable_offset), arg| match arg.type_ {
                    ArgType::ReadOnly => (readonly_offset + 1, writeable_offset),
                    ArgType::Writeable => (readonly_offset, writeable_offset + 1),
                },
            );

    // the Arg and its offset in its corresponding arg list
    let child_args: Vec<(Arg, usize)> = compute_levels
        .iter()
        .zip(child_physical_shapes.iter())
        .zip(child_buffer_idents.iter())
        .map(|((compute_level, physical_shape), buffer_ident)| {
            let physical_shape: Vec<Axis> = physical_shape
                .iter()
                .map(|Axis { addrs, kind }| Axis {
                    addrs: addrs
                        .iter()
                        .map(|addr| shape_addr_preference.get(&addr).copied().unwrap_or(*addr))
                        .collect(),
                    kind: *kind,
                })
                .collect();

            match *compute_level {
                // non-fusing
                0 => {
                    let offset = readonly_offset;
                    readonly_offset += 1;
                    (
                        Arg {
                            type_: ArgType::ReadOnly,
                            physical_shape: physical_shape,
                            ident: match buffer_ident {
                                Expr::Ident(s) if s.starts_with('s') => {
                                    Expr::ReadOnly(Box::new(buffer_ident.clone()))
                                }
                                _ => buffer_ident.clone(),
                            },
                        },
                        offset,
                    )
                }
                // fusing
                _ => {
                    let offset = writeable_offset;
                    writeable_offset += 1;
                    (
                        Arg {
                            type_: ArgType::Writeable,
                            physical_shape: physical_shape,
                            ident: buffer_ident.clone(),
                        },
                        offset,
                    )
                }
            }
        })
        .collect();

    // TODO prune loops and fold storage based on compute level
    //      to fold storage, you have to remove only dimensions that exist on the output (non-reduction dimensions)

    // for globalizing the shape addrs of remaining loop specs
    let globalize_loop_specs = |loop_spec: &LoopSpec| {
        let globalize_shape_addr = |addr| globalize_shape_addr(addr, &child_shape_addr_lists);
        LoopSpec {
            addrs: loop_spec.addrs.iter().map(globalize_shape_addr).collect(),
            axis: globalize_axis(&loop_spec.axis, &child_shape_addr_lists),
            ..loop_spec.clone()
        }
    };

    let loop_specs: Vec<LoopSpec> = loop_specs
        .into_iter()
        .filter(|spec| !prunable_axes.contains(&spec.axis))
        .map(globalize_loop_specs)
        .collect();

    // TODO can we do this better?
    addr_to_split_factor_list.extend(loop_specs.iter().flat_map(|spec| {
        spec.axis
            .addrs
            .iter()
            .map(|addr| (addr, &spec.split_factors))
    }));

    let child_buffer_bound_lists: Vec<Vec<Expr>> = child_physical_shapes
        .iter()
        .zip(child_args.iter())
        .map(|(child_physical_shape, (child_arg, offset))| {
            let arg_str = match child_arg.type_ {
                ArgType::ReadOnly => "inputs",
                ArgType::Writeable => "outputs",
            };
            (0..child_physical_shape.len())
                .map(|dim_ind| Expr::Indexed {
                    expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
                        expr: Box::new(Expr::Ident(arg_str.into())),
                        index: Box::new(Expr::Int(*offset)),
                    }))),
                    index: Box::new(Expr::Int(dim_ind)),
                })
                .collect()
        })
        .collect();

    let child_iter_ident_lists: Vec<Vec<Expr>> = child_physical_shapes
        .iter()
        .map(|child_physical_shape| {
            child_physical_shape
                .iter()
                .map(|Axis { addrs, kind }| {
                    let addr = addrs[0];
                    let addr = shape_addr_preference.get(&addr).copied().unwrap_or(addr);
                    let base_str = format!("i_{}_{}", addr.input_ind, addr.dim_ind);
                    Expr::Ident(match kind {
                        Bound::Base => base_str,
                        Bound::Split { ind, .. } => format!("{base_str}_{ind}"),
                    })
                })
                .collect()
        })
        .collect();

    let child_indexing_exprs: Vec<Expr> = child_buffer_bound_lists
        .iter()
        .zip(child_iter_ident_lists.iter())
        .map(|(bound_idents, iter_idents)| create_affine_index(&iter_idents, &bound_idents))
        .collect();

    let iter_idents: Vec<Expr> = physical_shape
        .iter()
        .map(|Axis { addrs, kind }| {
            let addr = addrs[0];
            let addr = child_shape_addr_lists[addr.input_ind][addr.dim_ind];
            let base_str = format!("i_{}_{}", addr.input_ind, addr.dim_ind);
            Expr::Ident(match kind {
                Bound::Base => base_str,
                Bound::Split { ind, .. } => format!("{base_str}_{ind}"),
            })
        })
        .collect();

    let buffer_bounds: Vec<Expr> = (0..physical_shape.len())
        .map(|dim_ind| Expr::Indexed {
            expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
                expr: Box::new(Expr::Ident("outputs".into())),
                index: Box::new(Expr::Int(writeable_offset)),
            }))),
            index: Box::new(Expr::Int(dim_ind)),
        })
        .collect();

    let indexing_expr: Expr = create_affine_index(&iter_idents, &buffer_bounds);

    let mut child_access_exprs: Vec<Expr> = child_args
        .iter()
        .zip(child_indexing_exprs)
        .map(|((arg, offset), indexing_expr)| make_access_expr(&arg, *offset, &indexing_expr))
        .collect();

    let physical_shape: Vec<Axis> = physical_shape
        .iter()
        .map(|axis| globalize_axis(axis, &child_shape_addr_lists))
        .collect();

    let arg = Arg {
        type_: ArgType::Writeable,
        physical_shape: physical_shape.clone(),
        ident: buffer_ident.clone(),
    };

    let access_expr: Expr = make_access_expr(&arg, writeable_offset, &indexing_expr);

    args.extend(child_args);
    if prunable_axes.is_empty() {
        args.push((arg, writeable_offset));
    }

    if child_access_exprs.len() == 1 {
        match op {
            '+' | '*' | '>' | '<' => child_access_exprs.insert(0, access_expr.clone()),
            '-' => child_access_exprs.insert(0, Expr::Int(0)),
            '/' => child_access_exprs.insert(0, Expr::Int(1)),
            _ => (),
        }
    }

    let op_statement = Statement::Assignment {
        left: access_expr,
        right: Expr::Op {
            op: *op,
            inputs: child_access_exprs,
        },
    };

    // create library function
    let function_fragment = build_library_function(
        &args,
        &loop_specs,
        &child_fragments,
        &compute_levels,
        &op_statement,
    );

    let mut fused_fragment = Block::default();
    if prunable_axes.is_empty() {
        // create library "kernel" function
        library.statements.push(Statement::Function {
            signature: FunctionSignature::Kernel(library_function_ident.clone()),
            body: function_fragment,
        });

        // create call site
        exec_block.statements.push(Statement::Call {
            ident: library_function_ident,
            in_args: args
                .iter()
                .filter(|(arg, _)| matches!(arg.type_, ArgType::ReadOnly))
                .map(|(arg, _)| arg.ident.clone())
                .collect(),
            out_args: args
                .iter()
                .filter(|(arg, _)| matches!(arg.type_, ArgType::Writeable))
                .map(|(arg, _)| arg.ident.clone())
                .collect(),
        });
    } else {
        fused_fragment
            .statements
            .extend(function_fragment.statements);
    }

    // TODO create drop

    *topo_ind += 1;

    (shape_addrs, physical_shape, buffer_ident, fused_fragment)
}

/// Convert `Axis`'s `ShapeAddr` from local (per-node) to global (per-graph)
fn globalize_axis(axis: &Axis, child_shape_addr_lists: &Vec<Vec<ShapeAddr>>) -> Axis {
    Axis {
        addrs: axis
            .addrs
            .iter()
            .map(|addr| globalize_shape_addr(addr, child_shape_addr_lists))
            .collect(),
        kind: axis.kind,
    }
}

/// Convert from local (per-node) `ShapeAddr` to global (per-graph) `ShapeAddr`
fn globalize_shape_addr(
    addr: &ShapeAddr,
    child_shape_addr_lists: &Vec<Vec<ShapeAddr>>,
) -> ShapeAddr {
    child_shape_addr_lists[addr.input_ind][addr.dim_ind]
}

/// Build up the access Expr, e.g., `outputs[offset].data[affine_indexing_expr]`
fn make_access_expr(arg: &Arg, offset: usize, indexing_expr: &Expr) -> Expr {
    let arg_str = match arg.type_ {
        ArgType::ReadOnly => "inputs",
        ArgType::Writeable => "outputs",
    };
    Expr::Indexed {
        expr: Box::new(Expr::DataOf(Box::new(Expr::Indexed {
            expr: Box::new(Expr::Ident(arg_str.into())),
            index: Box::new(Expr::Int(offset)),
        }))),
        index: Box::new(indexing_expr.clone()),
    }
}

/// Determines which loops should be pruned in the recursive call, specified by (dimension
/// index, Bound)
fn get_prunable_axes(
    loop_specs: &Vec<LoopSpec>,
    compute_level: usize,
    child_ind: usize,
) -> HashSet<Axis> {
    loop_specs
        .iter()
        .take(compute_level)
        .filter_map(|spec| {
            let addrs: Vec<ShapeAddr> = spec
                .axis
                .addrs
                .iter()
                .filter(|a| a.input_ind == child_ind)
                .cloned()
                .collect();

            (!addrs.is_empty()).then(|| Axis {
                addrs,
                kind: spec.bound,
            })
        })
        .collect()
}

/// `Expr`s for the dims of the buffer accounting for splitting
fn build_buffer_shape_exprs(
    physical_shape: &Vec<Axis>,
    split_factor_lists: &Vec<Vec<usize>>,
    child_shape_addrs: &Vec<Vec<ShapeAddr>>,
    addr_to_split_factor_list: &HashMap<&ShapeAddr, &Vec<usize>>,
) -> Vec<Expr> {
    let exprs: Vec<Expr> = physical_shape
        .iter()
        .map(|axis| {
            let addr = axis.addrs[0];
            let base_shape_expr = input_shape_expr(&addr);
            let factors = addr_to_split_factor_list[&addr];
            match &axis.kind {
                Bound::Base => base_shape_expr,
                Bound::Split { factor, ind: 0 } => {
                    create_split_bound_expr(base_shape_expr, factors)
                }
                Bound::Split { factor, ind } => Expr::Int(*factor),
            }
        })
        .collect();
    match exprs.is_empty() {
        true => vec![Expr::Int(1)],
        false => exprs,
    }
}

fn input_shape_expr(addr: &ShapeAddr) -> Expr {
    Expr::Indexed {
        expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
            expr: Box::new(Expr::Ident("inputs".into())),
            index: Box::new(Expr::Int(addr.input_ind)),
        }))),
        index: Box::new(Expr::Int(addr.dim_ind)),
    }
}

fn build_library_function(
    args: &Vec<(Arg, usize)>,
    loop_specs: &Vec<LoopSpec>,
    child_fragments: &Vec<Block>,
    compute_levels: &Vec<usize>,
    op_statement: &Statement,
) -> Block {
    // value: (arg, offset, dim_ind)
    let axis_to_arg_info: HashMap<Axis, (Arg, usize, usize)> = args
        .iter()
        .rev()
        .flat_map(|(arg, offset)| {
            arg.physical_shape
                .iter()
                .enumerate()
                .flat_map(move |(dim_ind, axis)| {
                    axis.addrs.iter().copied().map(move |addr| {
                        (
                            Axis {
                                addrs: vec![addr],
                                kind: axis.kind,
                            },
                            (arg.clone(), *offset, dim_ind),
                        )
                    })
                })
        })
        .collect();

    fn loop_bound_from_arg_info(arg: &Arg, offset: usize, dim_ind: usize) -> Expr {
        let ident = match arg.type_ {
            ArgType::ReadOnly => "inputs",
            ArgType::Writeable => "outputs",
        };
        Expr::Indexed {
            expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
                expr: Box::new(Expr::Ident(ident.into())),
                index: Box::new(Expr::Int(offset)),
            }))),
            index: Box::new(Expr::Int(dim_ind)),
        }
    }

    let axis_to_bound = |axis: &Axis| match axis_to_arg_info.get(&axis) {
        Some((arg, offset, dim_ind)) => Some(loop_bound_from_arg_info(arg, *offset, *dim_ind)),
        None => None,
    };

    let get_bound = |spec: &LoopSpec| {
        // TODO prefer factor over indexed shape
        let bound = axis_to_bound(&Axis {
            addrs: vec![spec.axis.addrs[0]],
            kind: spec.axis.kind,
        });

        bound.unwrap_or_else(|| {
            match &spec.axis.kind {
                Bound::Base => todo!(), // TODO look for split0 and reconstruct
                Bound::Split { factor, ind } => match ind {
                    0 => {
                        let base_bound = axis_to_bound(&Axis {
                            addrs: vec![spec.axis.addrs[0]],
                            kind: Bound::Base,
                        })
                        .expect("Could not compute split or base bound.");
                        create_split_bound_expr(base_bound, &spec.split_factors)
                    }
                    _ => Expr::Int(*factor),
                },
            }
        })
    };

    let get_base_bound = |spec: &LoopSpec| match &spec.axis.kind {
        Bound::Base | Bound::Split { ind: 0, .. } => Some(get_bound(spec)),
        Bound::Split { factor, ind } => axis_to_bound(&Axis {
            kind: Bound::Base,
            addrs: vec![spec.axis.addrs[0]],
        }),
    };

    let make_empty_loop = |spec: &LoopSpec| {
        let axis = &spec.axis;
        let split_factors = &spec.split_factors;

        let base_index_str = format!("i_{}_{}", axis.addrs[0].input_ind, axis.addrs[0].dim_ind);

        let bound = get_bound(&spec);

        let statements = if let Some(split_factors) = &spec.index_reconstruction {
            let base_bound = get_base_bound(&spec);
            // TODO find some way to still guard here--probably passing original shape as metadata
            match base_bound {
                Some(base_bound) => create_index_reconstruction_statements(
                    &Expr::Ident(base_index_str.clone()),
                    &base_bound,
                    split_factors,
                ),
                None => vec![],
            }
        } else {
            Vec::new()
        };

        let index: Expr = Expr::Ident(match axis.kind {
            Bound::Base => base_index_str,
            Bound::Split { ind, .. } => format!("{base_index_str}_{ind}"),
        });

        Statement::Loop {
            index,
            bound,
            body: Block { statements },
            parallel: true,
        }
    };

    // perform loop fusions
    let mut loops: Vec<Statement> = loop_specs.iter().map(make_empty_loop).collect();
    for (child_ind, compute_level) in compute_levels.iter().enumerate() {
        if *compute_level > 0 {
            if let Statement::Loop { ref mut body, .. } = loops[*compute_level - 1] {
                body.statements
                    .extend(child_fragments[child_ind].statements.clone());
            } else {
                panic!("Loop `Statement` not of variant `Loop`.")
            }
        }
    }

    let loop_stack: Statement =
        loops
            .into_iter()
            .rev()
            .fold(op_statement.clone(), |loop_stack, mut loop_| {
                if let Statement::Loop { ref mut body, .. } = loop_ {
                    body.statements.push(loop_stack);
                }
                loop_
            });

    Block {
        statements: vec![loop_stack],
    }
}

/// Get IR for `count` function
fn count(count: usize) -> Statement {
    Statement::Function {
        signature: FunctionSignature::Count,
        body: Block {
            statements: vec![Statement::Return {
                value: Expr::Int(count),
            }],
        },
    }
}

/// Get IR for `ranks` function
fn ranks(ranks: Vec<usize>) -> Statement {
    Statement::Function {
        signature: FunctionSignature::Ranks,
        body: Block {
            statements: ranks
                .iter()
                .enumerate()
                .map(|(ind, rank)| Statement::Assignment {
                    left: Expr::Indexed {
                        expr: Box::new(Expr::Ident("ranks".into())),
                        index: Box::new(Expr::Int(ind)),
                    },
                    right: Expr::Int(*rank),
                })
                .collect(),
        },
    }
}

/// Get IR for `shapes` function
fn shapes(shapes: Vec<Vec<Expr>>) -> Statement {
    Statement::Function {
        signature: FunctionSignature::Shapes,
        body: Block {
            statements: shapes
                .iter()
                .enumerate()
                .flat_map(|(shape_ind, shape)| {
                    shape.iter().enumerate().map(move |(dim_ind, dim_expr)| {
                        Statement::Assignment {
                            left: Expr::Indexed {
                                expr: Box::new(Expr::Indexed {
                                    expr: Box::new(Expr::Ident("shapes".into())),
                                    index: Box::new(Expr::Int(shape_ind)),
                                }),
                                index: Box::new(Expr::Int(dim_ind)),
                            },
                            right: dim_expr.clone(), // TODO avoid clone by taking ownership of input?
                        }
                    })
                })
                .collect(),
        },
    }
}

// TODO: take split_factors: &Vec<Expr> instead of mapping in this function
fn create_split_bound_expr(buffer_ident: Expr, split_factors: &Vec<usize>) -> Expr {
    let tile_width_expr = Expr::Op {
        op: '*',
        inputs: split_factors
            .iter()
            .map(|factor| Expr::Int(*factor))
            .collect(),
    };

    let numerator = Expr::Op {
        op: '-',
        inputs: vec![
            Expr::Op {
                op: '+',
                inputs: vec![buffer_ident, tile_width_expr.clone()],
            },
            Expr::Int(1),
        ],
    };

    Expr::Op {
        op: '/',
        inputs: vec![numerator, tile_width_expr],
    }
}

fn create_index_reconstruction_statements(
    base_iterator_ident: &Expr,
    base_bound_ident: &Expr,
    split_factors: &Vec<usize>,
) -> Vec<Statement> {
    let Expr::Ident(base_iterator_string) = base_iterator_ident else {
        panic!("Got non-Ident base ident.")
    };
    let iterators: Vec<Expr> = (0..=split_factors.len())
        .map(|ind| Expr::Ident(format!("{}_{}", base_iterator_string, ind)))
        .collect();

    let mut weights: Vec<Expr> = Vec::with_capacity(split_factors.len() + 1);
    let mut acc: usize = split_factors.iter().product();
    weights.push(Expr::Int(acc));
    for f in split_factors.iter().take(split_factors.len()) {
        acc /= *f;
        weights.push(Expr::Int(acc));
    }

    let reconstructed_index = Expr::Op {
        op: '+',
        inputs: iterators
            .into_iter()
            .zip(weights.into_iter())
            .map(|(it, w)| Expr::Op {
                op: '*',
                inputs: vec![w, it],
            })
            .collect(),
    };

    vec![
        Statement::Declaration {
            ident: base_iterator_ident.clone(),
            value: reconstructed_index,
            type_: Type::Int(false),
        },
        Statement::Skip {
            index: base_iterator_ident.clone(),
            bound: base_bound_ident.clone(),
        },
    ]
}

fn shape_addrs_to_indexing_expr(shape_addrs: &Vec<ShapeAddr>) -> Expr {
    let (iter_ident, bound_ident): (Vec<Expr>, Vec<Expr>) = shape_addrs
        .iter()
        .map(|ShapeAddr { input_ind, dim_ind }| {
            (
                Expr::Ident(format!("i_{}_{}", input_ind, dim_ind)),
                Expr::Ident(format!("b_{}_{}", input_ind, dim_ind)),
            )
        })
        .unzip();

    create_affine_index(&iter_ident, &bound_ident)
}

fn create_affine_index(indices: &Vec<Expr>, bounds: &Vec<Expr>) -> Expr {
    if bounds.len() == 1 && bounds[0] == Expr::Int(1) {
        return Expr::Int(0);
    }

    let indices = bounds
        .iter()
        .rev()
        .zip(indices.iter().rev())
        .map(|(_bound, index)| index)
        .rev()
        .collect::<Vec<_>>();
    let d = indices.len();
    let mut sum_expr = None;
    for k in 0..d {
        let mut product_expr = None;
        for m in (k + 1)..d {
            product_expr = Some(match product_expr {
                Some(expr) => Expr::Op {
                    op: '*',
                    inputs: vec![expr, bounds[m].clone()],
                },
                None => bounds[m].clone(),
            });
        }
        let partial_expr = match product_expr {
            Some(expr) => Expr::Op {
                op: '*',
                inputs: vec![indices[k].clone(), expr],
            },
            None => indices[k].clone(),
        };
        sum_expr = Some(match sum_expr {
            Some(expr) => Expr::Op {
                op: '+',
                inputs: vec![expr, partial_expr],
            },
            None => partial_expr,
        });
    }
    sum_expr.unwrap_or(Expr::Int(0)) // Return 0 if no indices are provided
}

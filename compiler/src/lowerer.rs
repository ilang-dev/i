use std::collections::{HashMap, HashSet};

use crate::block::{Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Axis, Bound, Graph, LoopSpec, Node, NodeBody, ShapeAddr};

#[derive(Clone, Debug)]
enum Arg {
    ReadOnly(usize),
    Writeable(usize),
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
        .leaves()
        .iter()
        .enumerate()
        .map(|(ind, node)| (node.lock().unwrap().id, ind))
        .collect();
    let mut shape_addr_preference = HashMap::<ShapeAddr, ShapeAddr>::new();

    let lowereds: Vec<(usize, Vec<ShapeAddr>, Vec<Axis>, Expr, Block)> = graph
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
                &mut shape_addr_preference,
                &HashSet::new(),
                &mut vec![],
                &mut vec![],
            )
        })
        .collect();

    let shape_exprs: Vec<Vec<Expr>> = lowereds
        .iter()
        .map(|lowered| lowered.1.clone())
        .map(|shape_addr_list| shape_addr_list.iter().map(input_shape_expr).collect())
        .collect();

    Program {
        count: count(graph.roots().len()),
        ranks: ranks(lowereds.iter().map(|l| l.0).collect()),
        shapes: shapes(shape_exprs),
        library: library,
        exec: Statement::Function {
            signature: FunctionSignature::Exec,
            body: exec_block,
        },
    }
}

/// Lower node. Update library and exec block, return (rank, semantic shape, physical shape, buffer ident, fused fragment)
fn lower_node(
    node: &Node,
    root_ind: Option<usize>,
    topo_ind: &mut usize,
    library: &mut Block,
    exec_block: &mut Block,
    node_to_leaf_ind: &mut HashMap<usize, usize>,
    shape_addr_preference: &mut HashMap<ShapeAddr, ShapeAddr>,
    prunable_axes: &HashSet<Axis>,
    readonly_buffer_idents: &mut Vec<Expr>, // (ident for call site)
    writeable_buffer_idents: &mut Vec<Expr>, // (ident for call site)
) -> (usize, Vec<ShapeAddr>, Vec<Axis>, Expr, Block) {
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
        let rank = node.index.len();
        let logical_shape: Vec<ShapeAddr> = (0..node.index.len())
            .map(|dim_ind| ShapeAddr {
                input_ind: leaf_ind,
                dim_ind: dim_ind,
            })
            .collect();

        let physical_shape: Vec<Axis> = logical_shape
            .iter()
            .map(|addr| Axis {
                addr: *addr,
                kind: Bound::Base,
            })
            .collect();

        let buffer_ident = Expr::Indexed {
            expr: Box::new(Expr::Ident("inputs".into())),
            index: Box::new(Expr::Int(leaf_ind)),
        };

        return (
            rank,
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

    let (readonly_buffer_idents, writeable_buffer_idents) = match prunable_axes.is_empty() {
        true => (&mut vec![], &mut vec![]), // non-fused -> new arg lists
        false => (readonly_buffer_idents, writeable_buffer_idents), // fused -> use existing lists
    };

    let children_lowereds: Vec<(usize, Vec<ShapeAddr>, Vec<Axis>, Expr, Block)> = node
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
                &get_prunable_axes(&loop_specs, compute_level, child_ind),
                readonly_buffer_idents,
                writeable_buffer_idents,
            )
        })
        .collect();

    let child_shape_addr_lists: Vec<Vec<ShapeAddr>> =
        children_lowereds.iter().map(|l| l.1.clone()).collect();
    let child_physical_shapes: Vec<Vec<Axis>> =
        children_lowereds.iter().map(|l| l.2.clone()).collect();
    let child_buffer_idents: Vec<Expr> = children_lowereds.iter().map(|l| l.3.clone()).collect();
    let child_fragments: Vec<Block> = children_lowereds.iter().map(|l| l.4.clone()).collect();

    // update shape addr prefs
    for list in shape_addr_lists {
        let globalize = |addr: &ShapeAddr| globalize_shape_addr(*addr, &child_shape_addr_lists);
        let list: Vec<ShapeAddr> = list.iter().map(globalize).collect();
        for i in 0..list.len() {
            shape_addr_preference.entry(list[i]).or_insert(list[0]);
        }
    }

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
        .map(|Axis { addr, kind }| Axis {
            addr: shape_addrs[addr.dim_ind],
            kind: *kind,
        })
        .collect();

    // TODO we will also need the physical shapes for the kernel args at some point
    // physical shape of the buffer allocated for the current node
    let physical_shape: Vec<Axis> = physical_shape
        .iter()
        .filter(|axis| !prunable_axes.contains(axis))
        .map(|Axis { addr, kind }| Axis {
            addr: child_shape_addr_lists[addr.input_ind][addr.dim_ind],
            kind: *kind,
        })
        .collect();

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
                &physical_shape,
                &split_factor_lists,
                &child_shape_addr_lists,
                &addr_to_split_factor_list,
            ),
        });
    }

    // TODO prune loops and fold storage based on compute level
    //      to fold storage, you have to remove only dimensions that exist on the output (non-reduction dimensions)

    // for globalizing the shape addrs of remaining loop specs
    let globalize_loop_specs = |loop_spec: &LoopSpec| {
        let globalize = |&addr| globalize_shape_addr(addr, &child_shape_addr_lists);
        LoopSpec {
            addrs: loop_spec.addrs.iter().map(globalize).collect(),
            axis: Axis {
                addr: globalize(&loop_spec.axis.addr),
                kind: loop_spec.axis.kind,
            },
            ..loop_spec.clone()
        }
    };

    let loop_specs: Vec<LoopSpec> = loop_specs
        .into_iter()
        .filter(|spec| !prunable_axes.contains(&spec.axis))
        .map(globalize_loop_specs)
        .collect();

    // where the library function must start accessing
    let mut readonly_args_offset = readonly_buffer_idents.len();

    readonly_buffer_idents.extend(child_buffer_idents.iter().zip(compute_levels).filter_map(
        |(child_buffer_ident, compute_level)| match compute_level {
            // add only non-fused children
            0 => Some(match child_buffer_ident {
                Expr::Ident(s) if s.starts_with('s') => {
                    Expr::ReadOnly(Box::new(child_buffer_ident.clone()))
                }
                _ => child_buffer_ident.clone(),
            }),
            _ => None,
        },
    ));
    writeable_buffer_idents.push(buffer_ident.clone());

    let n_fused_nodes = child_buffer_idents
        .iter()
        .zip(compute_levels.iter())
        .filter(|(_, &compute_level)| compute_level > 0)
        .count();

    // where the library function must start accessing
    let mut writeable_args_offset = writeable_buffer_idents.len() - 1 - n_fused_nodes;

    let child_args: Vec<Arg> = compute_levels
        .iter()
        .map(|compute_level| match *compute_level {
            // non-fusing
            0 => {
                let offset = readonly_args_offset;
                readonly_args_offset += 1;
                Arg::ReadOnly(offset)
            }
            // fusing
            _ => {
                let offset = writeable_args_offset;
                writeable_args_offset += 1;
                Arg::Writeable(offset)
            }
        })
        .collect();

    let mut seen = HashSet::new();
    let bound_decl_statements: Vec<Statement> = child_shape_addr_lists
        .iter()
        .enumerate()
        .flat_map(|(child_ind, child_shape_addr_list)| {
            let (arg_str, offset) = match child_args[child_ind] {
                Arg::ReadOnly(offset) => ("inputs", offset),
                Arg::Writeable(offset) => ("outputs", offset),
            };
            child_shape_addr_list
                .iter()
                .map(move |child_shape_addr| (arg_str, offset, child_shape_addr))
        })
        .filter_map(|(arg_str, offset, child_shape_addr)| {
            let key = (child_shape_addr.input_ind, child_shape_addr.dim_ind);
            if !seen.insert(key) {
                return None;
            }

            Some(Statement::Declaration {
                ident: Expr::Ident(format!(
                    "b_{}_{}",
                    child_shape_addr.input_ind, child_shape_addr.dim_ind
                )),
                value: Expr::Indexed {
                    expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
                        expr: Box::new(Expr::Ident(arg_str.into())),
                        index: Box::new(Expr::Int(offset)),
                    }))),
                    index: Box::new(Expr::Int(child_shape_addr.dim_ind)),
                },
                type_: Type::Int(false),
            })
        })
        .collect();

    let bound_decl_block = match prunable_axes.is_empty() {
        true => Block {
            statements: bound_decl_statements,
        },
        false => Block::default(),
    };

    // TODO can we do this better?
    addr_to_split_factor_list.extend(
        loop_specs
            .iter()
            .map(|spec| (&spec.axis.addr, &spec.split_factors)),
    );

    let child_indexing_exprs: Vec<Expr> = child_physical_shapes
        .iter()
        .map(|child_physical_shape| {
            let child_physical_shape = child_physical_shape
                .iter()
                .map(|Axis { addr, kind }| Axis {
                    addr: shape_addr_preference.get(addr).copied().unwrap_or(*addr),
                    kind: *kind,
                })
                .collect::<Vec<_>>();
            let (iter_idents, bound_idents): (Vec<Expr>, Vec<Expr>) =
                physical_shape_to_indexing_idents(
                    &child_physical_shape,
                    &addr_to_split_factor_list,
                );
            create_affine_index(&iter_idents, &bound_idents)
        })
        .collect();

    let (iter_idents, bound_idents): (Vec<Expr>, Vec<Expr>) =
        physical_shape_to_indexing_idents(&physical_shape, &addr_to_split_factor_list);
    let indexing_expr: Expr = create_affine_index(&iter_idents, &bound_idents);

    let mut child_access_exprs: Vec<Expr> = child_args
        .iter()
        .zip(child_indexing_exprs)
        .map(|(arg, indexing_expr)| make_access_expr(&arg, &indexing_expr))
        .collect();

    let access_expr: Expr =
        make_access_expr(&Arg::Writeable(writeable_args_offset), &indexing_expr);

    if child_access_exprs.len() == 1 {
        child_access_exprs.insert(
            0,
            match op {
                '+' | '*' | '>' | '<' => access_expr.clone(),
                '-' => Expr::Int(0),
                '/' => Expr::Int(1),
                op => panic!("{op} cannot have exactly 1 arg."),
            },
        );
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
        &loop_specs,
        &child_fragments,
        &compute_levels,
        &op_statement,
        bound_decl_block,
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
            in_args: readonly_buffer_idents.clone(),
            out_args: writeable_buffer_idents.clone(),
        });
    } else {
        fused_fragment
            .statements
            .extend(function_fragment.statements);
    }

    // TODO create drop

    *topo_ind += 1;

    (
        node.index.len(),
        shape_addrs,
        physical_shape,
        buffer_ident,
        fused_fragment,
    )
}

/// Convert from local (per-node) `ShapeAddr` to global (per-graph) `ShapeAddr`
fn globalize_shape_addr(
    addr: ShapeAddr,
    child_shape_addr_lists: &Vec<Vec<ShapeAddr>>,
) -> ShapeAddr {
    child_shape_addr_lists[addr.input_ind][addr.dim_ind]
}

/// Build up the access Expr, e.g., `outputs[offset].data[affine_indexing_expr]`
fn make_access_expr(arg: &Arg, indexing_expr: &Expr) -> Expr {
    let (arglist_str, offset) = match arg {
        Arg::ReadOnly(offset) => ("inputs", offset),
        Arg::Writeable(offset) => ("outputs", offset),
    };
    Expr::Indexed {
        expr: Box::new(Expr::DataOf(Box::new(Expr::Indexed {
            expr: Box::new(Expr::Ident(arglist_str.into())),
            index: Box::new(Expr::Int(*offset)),
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
            spec.addrs
                .iter()
                .find(|ShapeAddr { input_ind, .. }| *input_ind == child_ind)
                .map(|ShapeAddr { dim_ind, .. }| Axis {
                    addr: ShapeAddr {
                        input_ind: child_ind,
                        dim_ind: *dim_ind,
                    },
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
            let base_shape_expr = input_shape_expr(&axis.addr);
            let factors = addr_to_split_factor_list[&axis.addr];
            match &axis.kind {
                Bound::Base => match factors.is_empty() {
                    false => create_split_bound_expr(base_shape_expr, factors),
                    true => base_shape_expr,
                },
                Bound::Factor(factor) => Expr::Int(factors[*factor - 1]),
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

fn physical_shape_to_indexing_idents(
    physical_shape: &Vec<Axis>,
    addr_to_split_factor_list: &HashMap<&ShapeAddr, &Vec<usize>>,
) -> (Vec<Expr>, Vec<Expr>) {
    physical_shape
        .iter()
        .map(|axis| {
            let split_factors = addr_to_split_factor_list[&axis.addr];
            let iter_ident = match axis.kind {
                Bound::Base => match split_factors.is_empty() {
                    true => Expr::Ident(format!("i_{}_{}", axis.addr.input_ind, axis.addr.dim_ind)),
                    false => {
                        Expr::Ident(format!("i_{}_{}_0", axis.addr.input_ind, axis.addr.dim_ind))
                    }
                },
                Bound::Factor(factor) => Expr::Ident(format!(
                    "i_{}_{}_{}",
                    axis.addr.input_ind, axis.addr.dim_ind, factor
                )),
            };
            let bound_ident = match axis.kind {
                Bound::Base => {
                    Expr::Ident(format!("b_{}_{}", axis.addr.input_ind, axis.addr.dim_ind))
                }
                Bound::Factor(factor) => Expr::Int(split_factors[factor - 1]),
            };
            (iter_ident, bound_ident)
        })
        .unzip()
}

fn build_library_function(
    loop_specs: &Vec<LoopSpec>,
    child_fragments: &Vec<Block>,
    compute_levels: &Vec<usize>,
    op_statement: &Statement,
    preamble: Block,
) -> Block {
    let make_empty_loop = |spec: &LoopSpec| {
        let ShapeAddr { input_ind, dim_ind } = spec.addrs[0]; // any addr works, default to 0-th
        let split_factors = &spec.split_factors;

        let base_bound_string = format!("b_{input_ind}_{dim_ind}");
        let base_index_string = format!("i_{input_ind}_{dim_ind}");

        let statements = if let Some(split_factors) = &spec.index_reconstruction {
            create_index_reconstruction_statements(
                &Expr::Ident(base_index_string.clone()),
                &Expr::Ident(base_bound_string.clone()),
                split_factors,
            )
        } else {
            Vec::new()
        };

        let (bound, index) = match &spec.bound {
            Bound::Base => {
                if split_factors.is_empty() {
                    (
                        Expr::Ident(base_bound_string),
                        Expr::Ident(base_index_string),
                    )
                } else {
                    (
                        create_split_bound_expr(Expr::Ident(base_bound_string), split_factors),
                        Expr::Ident(format!("{base_index_string}_0")),
                    )
                }
            }
            Bound::Factor(ind) => (
                Expr::Int(split_factors[*ind - 1]),
                Expr::Ident(format!("{base_index_string}_{ind}")),
            ),
        };

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
        statements: [preamble.statements, vec![loop_stack]].concat(),
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

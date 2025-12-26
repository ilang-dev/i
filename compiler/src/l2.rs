use std::collections::{HashMap, HashSet};

use crate::ast::Schedule;
use crate::block::{Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Bound, Graph, LoopSpec, Node, NodeBody, ShapeAddr};

// This function is responsible for the rank, shape, and exec functions.
// `rank` and `shape` are easy, but `exec` has some complexity. The API should
// be `void exec(const Tensor* inputs, size_t n_inputs, TensorMut* output)`. It
// must map any relevant input values to idents (tensors and dims), perform any
// allocations (and eventually frees), and launch kernels
pub fn lower(graph: &Graph) -> Program {
    // TODO what needs to be passed up from the recursion?
    //      - rank value
    //      - whatever necessary to populate `shapes`
    //      - whatever necessary to populate `exec`

    // every interior node has these things:
    // - allocation
    // - kernel/kernel fragment
    // - call site (except when fused)

    // freeing can be done only after all instances of the intermediate data are used

    // things that need to be reused between uses of nodes:
    // - buffer name
    // - [actually not this] kernel call site

    let mut library = Block::default();
    let mut exec_block = Block::default();
    let mut node_to_leaf_ind: HashMap<usize, usize> = HashMap::new();

    node_to_leaf_ind = graph
        .leaves()
        .iter()
        .enumerate()
        .map(|(ind, node)| (node.lock().unwrap().id, ind))
        .collect();

    // TODO put this note in the appropriate place, or otherwise document
    // "shapes" are encoded as (usize, usize) "addresses" pointing to
    // (input index, dimension index)

    let mut topo_ind = 0;

    let lowereds: Vec<(usize, Vec<ShapeAddr>, Expr, Expr, Block)> = graph
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
                vec![],
            )
        })
        .collect();

    // TODO this is duplicated from below and it would be nice if that weren't true
    let shape_addrs: Vec<Vec<ShapeAddr>> = lowereds.iter().map(|l| l.1.clone()).collect();
    let shape_exprs: Vec<Vec<Expr>> = shape_addrs
        .iter()
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

/// Lower node. Update library, return (rank, shape addr expr, indexing expr, fused fragment)
fn lower_node(
    node: &Node,
    root_ind: Option<usize>,
    topo_ind: &mut usize,
    library: &mut Block,
    exec_block: &mut Block,
    node_to_leaf_ind: &mut HashMap<usize, usize>,
    prunable_loops: Vec<(usize, Bound)>,
) -> (usize, Vec<ShapeAddr>, Expr, Expr, Block) {
    let NodeBody::Interior {
        op,
        shape_addr_lists,
        split_factor_lists,
        loop_specs,
        compute_levels,
    } = &node.body
    else {
        // handle leaf nodes
        let leaf_ind = node_to_leaf_ind[&node.id];
        let rank = node.index.len();
        let shape_addr = (0..node.index.len())
            .map(|dim_ind| ShapeAddr {
                input_ind: leaf_ind,
                dim_ind: dim_ind,
            })
            .collect();

        let (index_exprs, bound_exprs): (Vec<Expr>, Vec<Expr>) = (0..node.index.len())
            .map(|dim_ind| {
                (
                    Expr::Ident(format!("i_{leaf_ind}_{dim_ind}")),
                    Expr::Ident(format!("b_{leaf_ind}_{dim_ind}")),
                )
            })
            .unzip();
        let indexing_expr = create_affine_index(&index_exprs, &bound_exprs);

        let buffer_ident = Expr::Indexed {
            expr: Box::new(Expr::Ident("inputs".into())),
            index: Box::new(Expr::Int(leaf_ind)),
        };

        return (
            rank,
            shape_addr,
            indexing_expr,
            buffer_ident,
            Block::default(),
        );
    };

    assert!(
        node.children().len() == compute_levels.len(),
        "Number of compute level specifications does not match number of children"
    );

    let children_lowereds: Vec<(usize, Vec<ShapeAddr>, Expr, Expr, Block)> = node
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
                get_prunable_loops(&loop_specs, compute_level, child_ind),
            )
        })
        .collect();

    let child_shape_addrs: Vec<Vec<ShapeAddr>> =
        children_lowereds.iter().map(|l| l.1.clone()).collect();
    let child_indexing_exprs: Vec<Expr> = children_lowereds.iter().map(|l| l.2.clone()).collect();
    let child_buffer_idents: Vec<Expr> = children_lowereds.iter().map(|l| l.3.clone()).collect();
    let child_fragments: Vec<Block> = children_lowereds.iter().map(|l| l.4.clone()).collect();

    let library_function_ident = Expr::Ident(format!("f{topo_ind}"));
    let buffer_ident = match root_ind {
        None => Expr::Ident(format!("s{topo_ind}")),
        Some(root_ind) => Expr::Indexed {
            expr: Box::new(Expr::Ident("outputs".into())),
            index: Box::new(Expr::Int(root_ind)),
        },
    };

    // create indexing expr
    let (index_exprs, bound_exprs): (Vec<Expr>, Vec<Expr>) = shape_addr_lists
        .iter()
        .map(|list| list[0]) // any shape addr works, default to 0-th
        .map(|addr| {
            let ShapeAddr { input_ind, dim_ind } = child_shape_addrs[addr.input_ind][addr.dim_ind];
            (
                Expr::Ident(format!("i_{}_{}", input_ind, dim_ind)),
                Expr::Ident(format!("b_{}_{}", input_ind, dim_ind)),
            )
        })
        .unzip();
    let indexing_expr = create_affine_index(&index_exprs, &bound_exprs);

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
                &shape_addr_lists,
                &split_factor_lists,
                &child_shape_addrs,
            ),
        });
    }

    // TODO prune loops and fold storage based on compute level
    //      to fold storage, you have to remove only dimensions that exist on the output (non-reduction dimensions)

    // for filtering prunable loops
    let prunable = |loop_spec: &&LoopSpec| {
        !prunable_loops.iter().any(|(dim_ind, pruned_bound)| {
            Some(dim_ind) == loop_spec.output_dim.as_ref() && loop_spec.bound == *pruned_bound
        })
    };

    // for globalizing the shape addrs of remaining loop specs
    let globalize_loop_specs = |loop_spec: &LoopSpec| {
        let globalize = |&ShapeAddr { input_ind, dim_ind }| child_shape_addrs[input_ind][dim_ind];
        LoopSpec {
            addrs: loop_spec.addrs.iter().map(globalize).collect(),
            ..loop_spec.clone()
        }
    };

    let loop_specs: Vec<LoopSpec> = loop_specs
        .into_iter()
        .filter(prunable)
        .map(globalize_loop_specs)
        .collect();

    // TODO the library function needs shape in terms of children

    // create library function
    let function_fragment = build_library_function(&loop_specs, &child_fragments, &compute_levels);

    let mut fused_fragment = Block::default();
    if prunable_loops.is_empty() {
        // create library "kernel" function
        library.statements.push(Statement::Function {
            signature: FunctionSignature::Kernel(library_function_ident.clone()),
            body: function_fragment,
        });

        // create call site
        exec_block.statements.push(Statement::Call {
            ident: library_function_ident,
            in_args: vec![], // NOTE we need to pass arrays of tensors, not dynamic args lists TODO
            out_args: vec![], // NOTE we need to pass arrays of tensors, not dynamic args lists TODO
        });
    } else {
        fused_fragment
            .statements
            .extend(function_fragment.statements);
    }

    // TODO create drop

    // `Expr`s for the dims of the node without regard to splitting or fusion
    let semantic_shape_exprs: Vec<ShapeAddr> = shape_addr_lists
        .iter()
        .map(|shape_addr_list| shape_addr_list[0]) // any shape addr works, default to 0-th
        .map(|ShapeAddr { input_ind, dim_ind }| child_shape_addrs[input_ind][dim_ind])
        .collect();

    *topo_ind += 1;

    // TODO I think the shape output here can be seen as tracking the semantic
    //      shape rather than the shape of the physical buffers. All this is used
    //      for is determining the output shape from the input shape which does
    //      depend on the intermediate buffer layout, but only the semantics of
    //      the expression. (probably write this down somewhere)
    (
        node.index.len(),
        semantic_shape_exprs,
        indexing_expr,
        buffer_ident,
        fused_fragment,
    )
}

/// Determines which loops should be pruned in the recursive call, specified by (dimension
/// index, Bound)
fn get_prunable_loops(
    loop_specs: &Vec<LoopSpec>,
    compute_level: usize,
    child_ind: usize,
) -> Vec<(usize, Bound)> {
    loop_specs
        .iter()
        .take(compute_level)
        .filter_map(|spec| {
            spec.addrs
                .iter()
                .find(|ShapeAddr { input_ind, .. }| *input_ind == child_ind)
                .map(|ShapeAddr { dim_ind, .. }| (*dim_ind, spec.bound))
        })
        .collect()
}

/// `Expr`s for the dims of the buffer accounting for splitting
fn build_buffer_shape_exprs(
    shape_addr_lists: &Vec<Vec<ShapeAddr>>,
    split_factor_lists: &Vec<Vec<usize>>,
    child_shape_addrs: &Vec<Vec<ShapeAddr>>,
) -> Vec<Expr> {
    shape_addr_lists
        .iter()
        .map(|list| list[0]) // any shape addr works, default to 0-th
        .zip(split_factor_lists.iter())
        .flat_map(|(addr, factors)| {
            let global_addr = child_shape_addrs[addr.input_ind][addr.dim_ind];
            let base_shape_expr = input_shape_expr(&global_addr);
            match factors.is_empty() {
                true => vec![base_shape_expr],
                false => std::iter::once(create_split_bound_expr(base_shape_expr, factors))
                    .chain(factors.iter().map(|factor| Expr::Int(*factor)))
                    .collect(),
            }
        })
        .collect()
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
    loop_specs: &Vec<LoopSpec>,
    child_fragments: &Vec<Block>,
    compute_levels: &Vec<usize>,
) -> Block {
    // TODO write a real scalar op here
    // We need to know how to identify and index the buffers
    let op_statement = Statement::Return {
        value: Expr::Int(0),
    };

    // TODO need offsets for indexing iterators and bounds to avoid collisions during fusions
    // EDIT maybe not actually, because we're deleting loops, not adding them. what about indexing
    //      though? the iterators idents need to line up at least. how do we do that?

    // TODO either define loop bound idents or inline shape exprs
    let make_empty_loop = |spec: &LoopSpec| {
        let group = spec.group;
        let ind = spec.ind;
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
                        create_split_bound_expr(
                            Expr::Ident(format!("{base_bound_string}_0")),
                            split_factors,
                        ),
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
            .fold(op_statement, |loop_stack, mut loop_| {
                if let Statement::Loop { ref mut body, .. } = loop_ {
                    body.statements.push(loop_stack);
                }
                loop_
            });

    Block {
        statements: vec![loop_stack],
    }
}

// NOTES ON LIBRARY FUNCTION LOWERING
// TODO maybe can map node ids to input/output indices
// TODO input/ouptut arg indices must be prepared for call site according to function body

// we could be lowering:
// - non-fused node which does not fuse
//   - build loops
//   - take children as inputs
//   -
// - non-fused node which does fuse
// - fused node which does not fuse
// - fused node which does fuse

// if node is fused:
// - adjust buffer allocation
// - don't write library function
// - somehow return fragment

// if node fuses:
// - get fused's fragment, write into function body
// - adjust call site according to fused's childrne

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

fn create_affine_index(indices: &Vec<Expr>, bounds: &Vec<Expr>) -> Expr {
    if bounds.len() == 1 && bounds[0] == Expr::Int(1) {
        return Expr::Int(0);
    }

    let indices = bounds
        .iter()
        .rev()
        .zip(indices.iter().rev())
        .map(|(bound, index)| index)
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

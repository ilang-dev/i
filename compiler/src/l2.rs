use std::collections::{HashMap, HashSet};

use crate::ast::Schedule;
use crate::block::{Arg, Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{BoundAddr, Graph, LoopSpec, Node, NodeBody};

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

    let lowereds: Vec<(Expr, usize, Vec<Expr>)> = graph
        .roots()
        .iter()
        .enumerate()
        .map(|(root_ind, node)| {
            lower_node(
                &node.lock().unwrap(),
                Some(root_ind),
                &mut library,
                &mut exec_block,
                &mut node_to_leaf_ind,
                0,
            )
        })
        .collect();

    Program {
        count: count(graph.roots().len()),
        ranks: ranks(lowereds.iter().map(|l| l.1).collect()),
        shapes: shapes(lowereds.into_iter().map(|l| l.2).collect()),
        library: library,
        exec: Statement::Function {
            signature: FunctionSignature::Exec,
            body: exec_block,
        },
    }
}

/// Lower node. Update library, return (buffer ident, rank, shape address)
fn lower_node(
    node: &Node,
    root_ind: Option<usize>,
    library: &mut Block,
    exec_block: &mut Block,
    node_to_leaf_ind: &mut HashMap<usize, usize>,
    compute_level: usize,
) -> (Expr, usize, Vec<Expr>) {
    let NodeBody::Interior {
        op,
        shape_addr_lists,
        split_factor_lists,
        loop_specs,
        compute_levels,
    } = &node.body
    else {
        // handle leaf nodes
        let input_ind = node_to_leaf_ind[&node.id];
        let store_ident = Expr::Indexed {
            expr: Box::new(Expr::Ident("inputs".into())),
            index: Box::new(Expr::Int(input_ind)),
        };
        let rank = node.index.len();
        let shape_addr = (0..node.index.len())
            .map(|dim_ind| Expr::Indexed {
                expr: Box::new(Expr::ShapeOf(Box::new(store_ident.clone()))),
                index: Box::new(Expr::Int(dim_ind)),
            })
            .collect();
        return (store_ident, rank, shape_addr);
    };

    assert!(
        node.children().len() == compute_levels.len(),
        "Number of compute level specifications does not match number of children"
    );

    let children_lowereds: Vec<(Expr, usize, Vec<Expr>)> = node
        .children()
        .iter()
        .zip(compute_levels.iter())
        .map(|((node, _), &compute_level)| {
            lower_node(
                &node,
                None,
                library,
                exec_block,
                node_to_leaf_ind,
                compute_level,
            )
        })
        .collect();
    let child_store_idents: Vec<Expr> = children_lowereds.iter().map(|l| l.0.clone()).collect();
    let child_shapes: Vec<Vec<Expr>> = children_lowereds.iter().map(|l| l.2.clone()).collect();

    let topo_ind = library.statements.len();
    let library_function_ident = Expr::Ident(format!("f{topo_ind}"));
    let buffer_ident = match root_ind {
        None => Expr::Ident(format!("s{topo_ind}")),
        Some(root_ind) => Expr::Indexed {
            expr: Box::new(Expr::Ident("outputs".into())),
            index: Box::new(Expr::Int(root_ind)),
        },
    };

    // TODO prune loops and fold storage based on compute level
    //      to fold storage, you have to remove only dimensions that exist on the output (non-reduction dimensions)

    // TODO the library function needs shape in terms of children

    let library_function = build_library_function(&loop_specs, &child_shapes);

    // create library function
    library.statements.push(Statement::Function {
        signature: FunctionSignature::Kernel(library_function_ident.clone()),
        body: library_function,
    });

    // `Expr`s for the dims of the buffer accounting for splitting
    // TODO account for fusion as well
    let buffer_shape_exprs: Vec<Expr> = shape_addr_lists
        .iter()
        .map(|list| list[0]) // any shape addr works, default to 0-th
        .zip(split_factor_lists.iter())
        .flat_map(|((input_ind, dim_ind), factors)| {
            let base_shape_expr = child_shapes[input_ind][dim_ind].clone();
            match factors.is_empty() {
                true => vec![base_shape_expr],
                false => std::iter::once(create_split_bound_expr(
                    child_shapes[input_ind][dim_ind].clone(),
                    factors,
                ))
                .chain(factors.iter().map(|factor| Expr::Int(*factor)))
                .collect(),
            }
        })
        .collect();

    // create allocation
    if root_ind.is_none() {
        exec_block.statements.push(Statement::Alloc {
            index: topo_ind,
            initial_value: Box::new(Expr::Scalar(match op {
                '>' => f32::NEG_INFINITY,
                '<' => f32::INFINITY,
                '*' => 1.,
                _ => 0.,
            })),
            shape: buffer_shape_exprs,
        });
    }

    // TODO this is messed up in the case of fusing nodes (they must adopt their fusings children)
    // TODO this is also messed up in general beacuse we need to pass arrays of tensors, not
    //      dynamic args lists
    // create call site
    exec_block.statements.push(Statement::Call {
        ident: library_function_ident,
        in_args: child_store_idents,
        out_args: vec![buffer_ident.clone()],
    });

    // TODO create drop

    // `Expr`s for the dims of the node without regard to splitting or fusion
    let semantic_shape_exprs = shape_addr_lists
        .iter()
        .map(|list| list[0]) // any shape addr works, default to 0-th
        .map(|(input_ind, dim_ind)| child_shapes[input_ind][dim_ind].clone())
        .collect();

    // TODO I think the shape output here can be seen as tracking the semantic
    //      shape rather than the shape of the physical buffers. All this is used
    //      for is determining the output shape from the input shape which does
    //      depend on the intermediate buffer layout, but only the semantics of
    //      the expression. (probably write this down somewhere)
    (buffer_ident, node.index.len(), semantic_shape_exprs)
}

fn build_library_function(loop_specs: &Vec<LoopSpec>, child_shapes: &Vec<Vec<Expr>>) -> Block {
    // TODO write a real scalar op here
    let op_statement = Statement::Return {
        value: Expr::Int(0),
    };

    // TODO either define loop bound idents or inline shape exprs
    let make_empty_loop = |spec: &LoopSpec| {
        let group = spec.group;
        let ind = spec.ind;

        let (bound, index) = match &spec.bound_addr {
            BoundAddr::Base {
                input_ind,
                dim_ind,
                split_factors,
            } => {
                if split_factors.is_empty() {
                    (
                        Expr::Ident(format!("b{group}")),
                        Expr::Ident(format!("i{group}")),
                    )
                } else {
                    (
                        create_split_bound_expr(
                            child_shapes[*input_ind][*dim_ind].clone(),
                            split_factors,
                        ),
                        Expr::Ident(format!("i{group}_0")),
                    )
                }
            }
            BoundAddr::Factor(factor) => {
                (Expr::Int(*factor), Expr::Ident(format!("i{group}_{ind}")))
            }
        };

        let statements = match &spec.index_reconstruction {
            Some(split_factors) => {
                create_index_reconstruction_statements(&index, &bound, split_factors)
            }
            None => Vec::new(),
        };

        Statement::Loop {
            index,
            bound,
            body: Block { statements },
            parallel: true,
        }
    };

    let loop_stack: Statement =
        loop_specs
            .iter()
            .map(make_empty_loop)
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

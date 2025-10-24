use std::collections::HashMap;

use crate::ast::Schedule;
use crate::block::{Arg, Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Graph, Node, NodeBody};

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
            )
        })
        .collect();

    Program {
        count: count(graph.roots().len()),
        ranks: ranks(&lowereds.iter().map(|l| l.1).collect()),
        shapes: shapes(&lowereds.iter().map(|l| l.2.clone()).collect()),
        library: library.clone(), // TODO can we avoid clone?
        exec: Statement::Function {
            signature: FunctionSignature::Exec,
            body: exec_block.clone(), // TODO can we avoid clone?
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
) -> (Expr, usize, Vec<Expr>) {
    let NodeBody::Interior {
        op,
        schedule,
        shape,
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

    let children_lowereds: Vec<(Expr, usize, Vec<Expr>)> = node
        .children()
        .iter()
        .map(|(node, _)| lower_node(&node, None, library, exec_block, node_to_leaf_ind))
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

    // TODO the library function needs shape in terms of children
    // TODO alloc needs shape in terms of inputs (Expr)

    // create library function
    library.statements.push(Statement::Function {
        signature: FunctionSignature::Kernel(library_function_ident.clone()),
        body: Block::default(), // TODO
    });

    let shape: Vec<_> = get_local_shape_addrs(node)
        .iter()
        .map(|(input_ind, dim_ind)| child_shapes[*input_ind][*dim_ind].clone())
        .collect();

    // create allocation
    if root_ind.is_none() {
        exec_block.statements.push(Statement::Declaration {
            ident: buffer_ident.clone(),
            value: Expr::Alloc {
                initial_value: Box::new(match op {
                    '>' => Expr::Scalar(f32::NEG_INFINITY),
                    '<' => Expr::Scalar(f32::INFINITY),
                    '*' => Expr::Scalar(1.),
                    _ => Expr::Scalar(0.),
                }),
                shape: shape.clone(), // TODO account for fusion
            },
            type_: Type::Array(true),
        });
    }

    // create call site
    exec_block.statements.push(Statement::Call {
        ident: library_function_ident,
        in_args: child_store_idents,
        out_args: vec![buffer_ident.clone()],
    });

    // TODO create drop

    // TODO I think the shape output here can be seen as tracking the semantic
    //      shape rather than the shape of the physical buffers. All this is used
    //      for is determining the output shape from the input shape which does
    //      depend on the intermediate buffer layout, but only the semantics of
    //      the expression. (probably write this down somewhere)
    (buffer_ident, node.index.len(), shape)
}

/// Compute shape address from index and child indices
/// written by ChatGPT
fn get_local_shape_addrs(node: &Node) -> Vec<(usize, usize)> {
    let child_indexes: Vec<String> = node
        .children()
        .iter()
        .map(|(_, child_ind)| child_ind.clone())
        .collect();
    let map: HashMap<char, (usize, usize)> = {
        let mut m = HashMap::new();
        for (i, s) in child_indexes.iter().enumerate() {
            for (d, c) in s.chars().enumerate() {
                m.entry(c).or_insert((i, d));
            }
        }
        m
    };
    node.index.chars().map(|c| *map.get(&c).unwrap()).collect()
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
fn ranks(ranks: &Vec<usize>) -> Statement {
    Statement::Function {
        signature: FunctionSignature::Ranks,
        body: Block {
            statements: ranks
                .iter()
                .enumerate()
                .map(|(ind, rank)| Statement::Assignment {
                    left: Expr::Indexed {
                        expr: Box::new(Expr::Ident("rank".into())),
                        index: Box::new(Expr::Int(ind)),
                    },
                    right: Expr::Int(*rank),
                })
                .collect(),
        },
    }
}

/// Get IR for `shapes` function
fn shapes(shapes: &Vec<Vec<Expr>>) -> Statement {
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
                                    expr: Box::new(Expr::Ident("shape".into())),
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

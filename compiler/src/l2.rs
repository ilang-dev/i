use std::collections::HashMap;

use crate::ast::Schedule;
use crate::block::{Arg, Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Graph, Node, NodeBody};

pub struct Lowerer; // TODO is this necessary?

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {}
    }

    // This function is responsible for the rank, shape, and exec functions.
    // `rank` and `shape` are easy, but `exec` has some complexity. The API should
    // be `void exec(const Tensor* inputs, size_t n_inputs, TensorMut* output)`. It
    // must map any relevant input values to idents (tensors and dims), perform any
    // allocations (and eventually frees), and launch kernels
    pub fn lower(&self, graph: &Graph) -> Program {
        // TODO remove `&self` arg

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

        let node_id_to_input_index: HashMap<usize, usize> = graph
            .leaves()
            .iter()
            .enumerate()
            .map(|(ind, node)| (node.lock().unwrap().id, ind))
            .collect();

        // TODO put this note in the appropriate place, or otherwise document
        // "shapes" are encoded as (usize, usize) "addresses" pointing to
        // (input index, dimension index)

        // TODO: can this really not be an iterator? this is so ugly
        let mut topo_inds: Vec<usize> = vec![];
        let mut ranks: Vec<usize> = vec![];
        let mut shapes: Vec<Vec<(usize, usize)>> = vec![];
        for node in graph.roots() {
            let (topo_ind, rank, shape) = Self::lower_node(
                &node.lock().unwrap(),
                &mut library,
                &mut exec_block,
                &node_id_to_input_index,
            );
            topo_inds.push(topo_ind);
            ranks.push(rank);
            shapes.push(shape);
        }
        //let (topo_inds, ranks, shapes): (Vec<usize>, Vec<usize>, Vec<Vec<usize>>) = graph
        //    .roots()
        //    .iter()
        //    .map(|node| Self::lower_node(&node.lock().unwrap(), &mut library))
        //    .unzip(); // unzip only works on 2-tuples

        // TODO convert from shape addressed to expressions
        let shapes = vec![vec![2, 2], vec![2, 2]];

        Program {
            count: Self::count(graph.roots().len()),
            ranks: Self::ranks(&ranks),
            shapes: Self::shapes(&shapes),
            library,
            exec: Statement::Function {
                signature: FunctionSignature::Exec,
                body: exec_block,
            },
        }
    }

    /// Lower node. Update library, return (topolical index, rank, shape address)
    fn lower_node(
        node: &Node,
        library: &mut Block,
        exec_block: &mut Block,
        node_id_to_input_index: &HashMap<usize, usize>,
    ) -> (usize, usize, Vec<(usize, usize)>) {
        let topo_ind = library.statements.len();
        let library_function_ident = format!("f{topo_ind}");
        let buffer_ident = format!("s{topo_ind}");

        let NodeBody::Interior {
            op,
            schedule,
            shape,
        } = &node.body
        else {
            return (
                topo_ind,
                node.index.len(),
                vec![(0, 0), (0, 1)], // TODO
                                      // get the input index, query its shape
                                      //(0..node.index.len())).map(|ind| format!("{}"))
            );
        };

        // obviously don't recurse on leaf nodes

        // TODO create library function
        //      this only requires children to be lowered first in fusion cases
        library.statements.push(Statement::Function {
            signature: FunctionSignature::Kernel(library_function_ident.clone()),
            body: Block::default(), // TODO
        });

        // create allocation
        if let NodeBody::Interior {
            op,
            schedule,
            shape,
        } = &node.body
        {
            exec_block.statements.push(Statement::Declaration {
                ident: buffer_ident.clone(),
                value: Expr::Alloc {
                    initial_value: Box::new(match op {
                        '>' => Expr::Scalar(f32::NEG_INFINITY),
                        '<' => Expr::Scalar(f32::INFINITY),
                        '*' => Expr::Scalar(1.),
                        _ => Expr::Scalar(0.),
                    }),
                    shape: vec![], // Vec<Expr>, // TODO
                },
                type_: Type::Array(true),
            });
        };

        // TODO create call site
        //      need to know child buffer idents
        //      need to know own buffer ident
        //      need kernel ident
        exec_block.statements.push(Statement::Call {
            ident: library_function_ident,
            in_args: vec![], // TODO idents of child buffers (having been lowered)
            out_args: vec![Expr::Ident(buffer_ident)],
        });

        // TODO create drop

        (
            topo_ind,
            node.index.len(),
            vec![(0, 0), (0, 1)], // TODO
        )
    }

    ///// Get allocation declaration
    //fn create_alloc_declaration(node: &Node, topo_ind: usize) -> Statement {
    //}

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
    fn shapes(shapes: &Vec<Vec<usize>>) -> Statement {
        Statement::Function {
            signature: FunctionSignature::Shapes,
            body: Block {
                statements: shapes
                    .iter()
                    .enumerate()
                    .flat_map(|(shape_ind, shape)| {
                        shape
                            .iter()
                            .enumerate()
                            .map(move |(dim_ind, dim)| Statement::Assignment {
                                left: Expr::Indexed {
                                    expr: Box::new(Expr::Indexed {
                                        expr: Box::new(Expr::Ident("shape".into())),
                                        index: Box::new(Expr::Int(shape_ind)),
                                    }),
                                    index: Box::new(Expr::Int(dim_ind)),
                                },
                                right: Expr::Int(*dim),
                            })
                    })
                    .collect(),
            },
        }
    }
}

use std::collections::HashMap;

use crate::ast::Schedule;
use crate::block::{Arg, Block, Expr, FunctionSignature, Program, Statement, Type};
use crate::graph::{Graph, Node, NodeBody};

pub struct Lowerer {
    library: Block,
    exec_block: Block,
    node_id_to_input_index: HashMap<usize, usize>,
} // TODO is this necessary?

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {
            library: Block::default(),
            exec_block: Block::default(),
            node_id_to_input_index: HashMap::new(),
        }
    }

    // This function is responsible for the rank, shape, and exec functions.
    // `rank` and `shape` are easy, but `exec` has some complexity. The API should
    // be `void exec(const Tensor* inputs, size_t n_inputs, TensorMut* output)`. It
    // must map any relevant input values to idents (tensors and dims), perform any
    // allocations (and eventually frees), and launch kernels
    pub fn lower(&mut self, graph: &Graph) -> Program {
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

        self.node_id_to_input_index = graph
            .leaves()
            .iter()
            .enumerate()
            .map(|(ind, node)| (node.lock().unwrap().id, ind))
            .collect();

        // TODO put this note in the appropriate place, or otherwise document
        // "shapes" are encoded as (usize, usize) "addresses" pointing to
        // (input index, dimension index)

        // TODO: can this really not be an iterator? this is so ugly
        let mut store_idents: Vec<Expr> = vec![];
        let mut ranks: Vec<usize> = vec![];
        let mut shapes: Vec<Vec<(usize, usize)>> = vec![];
        for node in graph.roots() {
            let (store_ident, rank, shape) = self.lower_node(&node.lock().unwrap());
            store_idents.push(store_ident);
            ranks.push(rank);
            shapes.push(shape);
        }

        Program {
            count: Self::count(graph.roots().len()),
            ranks: Self::ranks(&ranks),
            shapes: Self::shapes(&shapes),
            library: self.library.clone(), // TODO can we avoid clone?
            exec: Statement::Function {
                signature: FunctionSignature::Exec,
                body: self.exec_block.clone(), // TODO can we avoid clone?
            },
        }
    }

    /// Lower node. Update library, return (buffer ident, rank, shape address)
    fn lower_node(&mut self, node: &Node) -> (Expr, usize, Vec<(usize, usize)>) {
        let NodeBody::Interior {
            op,
            schedule,
            shape,
        } = &node.body
        else {
            // handle leaf nodes
            let input_ind = self.node_id_to_input_index[&node.id];
            let store_ident = Expr::Indexed {
                expr: Box::new(Expr::Ident("inputs".into())),
                index: Box::new(Expr::Int(input_ind)),
            };
            let rank = node.index.len();
            let shape_addr = (0..node.index.len())
                .map(|dim_ind| (input_ind, dim_ind))
                .collect();
            return (store_ident, rank, shape_addr);
        };

        // lower children
        let mut child_store_idents: Vec<Expr> = vec![];
        let mut child_ranks: Vec<usize> = vec![];
        let mut child_shapes: Vec<Vec<(usize, usize)>> = vec![];
        for (node, _) in node.children() {
            let (store_ident, rank, shape) = self.lower_node(&node);
            child_store_idents.push(store_ident);
            child_ranks.push(rank);
            child_shapes.push(shape);
        }

        let topo_ind = self.library.statements.len();
        let library_function_ident = format!("f{topo_ind}");
        let buffer_ident = format!("s{topo_ind}");

        let local_shape_addrs = Self::get_local_shape_addrs(node);

        // TODO the library function needs shape in terms of children
        // TODO alloc needs shape in terms of inputs (Expr)

        // create library function
        self.library.statements.push(Statement::Function {
            signature: FunctionSignature::Kernel(library_function_ident.clone()),
            body: Block::default(), // TODO
        });

        // create allocation
        self.exec_block.statements.push(Statement::Declaration {
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

        // create call site
        self.exec_block.statements.push(Statement::Call {
            ident: library_function_ident,
            in_args: child_store_idents,
            out_args: vec![Expr::Ident(buffer_ident.clone())],
        });

        // TODO create drop

        // TODO the shape returned must be in terms of the inputs

        (
            Expr::Ident(buffer_ident),
            node.index.len(),
            vec![(0, 0), (0, 1)], // TODO use child shape addrs; depends on fusion
        )
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
    fn shapes(shapes: &Vec<Vec<(usize, usize)>>) -> Statement {
        Statement::Function {
            signature: FunctionSignature::Shapes,
            body: Block {
                statements: shapes
                    .iter()
                    .enumerate()
                    .flat_map(|(shape_ind, shape)| {
                        shape
                            .iter()
                            .map(move |(input_ind, dim_ind)| Statement::Assignment {
                                left: Expr::Indexed {
                                    expr: Box::new(Expr::Indexed {
                                        expr: Box::new(Expr::Ident("shape".into())),
                                        index: Box::new(Expr::Int(shape_ind)),
                                    }),
                                    index: Box::new(Expr::Int(*dim_ind)),
                                },
                                right: Expr::Indexed {
                                    expr: Box::new(Expr::ShapeOf(Box::new(Expr::Indexed {
                                        expr: Box::new(Expr::Ident("inputs".into())),
                                        index: Box::new(Expr::Int(*input_ind)),
                                    }))),
                                    index: Box::new(Expr::Int(*dim_ind)),
                                },
                            })
                    })
                    .collect(),
            },
        }
    }
}

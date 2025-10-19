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

        let (ranks, shapes): (Vec<usize>, Vec<Vec<usize>>) = graph
            .roots()
            .iter()
            .map(|node| Self::lower_node(&node.lock().unwrap(), &mut library))
            .unzip();

        Program {
            count: Self::count(graph.roots().len()),
            ranks: Self::ranks(&ranks),
            shapes: Self::shapes(&shapes),
            library,
            exec: Statement::Function {
                signature: FunctionSignature::Exec,
                body: Block::default(),
            }, // TODO
        }
    }

    /// Lower node. Update library, return (rank, shape)
    fn lower_node(node: &Node, library: &mut Block) -> (usize, Vec<usize>) {
        (3, vec![3, 4, 5]) // TODO
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

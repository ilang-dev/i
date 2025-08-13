use std::collections::{HashMap, HashSet};

use crate::ast::Schedule;
use crate::block::{Arg, Block, Expr, Program, Statement, Type};
use crate::graph::{Graph, Node, NodeBody};

pub struct Lowerer {
    input_args: Vec<Arg>,
    input_array_counter: usize,
    base_loop_counter: usize,
    store_counter: usize,
    split_factor_count: usize,
}

#[derive(Clone, Debug)]
struct Lowered {
    def_block: Block,
    alloc_statements: HashMap<String, Statement>, // (ident, Statement::Alloc)
    exec_block: Block,
    def_args: Vec<Arg>, // only populated for kernel fragments, empty for full kernels
    loop_idents: HashMap<char, (String, String)>,
    store_ident: String,
    shape: Vec<(usize, usize)>,
}

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {
            input_args: Vec::new(),
            input_array_counter: 0,
            base_loop_counter: 0,
            store_counter: 0,
            split_factor_count: 0,
        }
    }

    fn get_char_indices(index: &String) -> Vec<char> {
        let mut seen = HashSet::new();
        index.chars().filter(|c| seen.insert(*c)).collect()
    }

    pub fn lower(&mut self, graph: &Graph) -> Program {
        assert_eq!(
            graph.roots().len(),
            1,
            "Attempted to lower `Graph` of {} roots.",
            graph.roots().len()
        );

        let mut memo = HashMap::<usize, Lowered>::new();
        let lowered = self.lower_node(
            &graph.root().lock().unwrap(),
            HashSet::new(),
            true,
            &mut memo,
        );

        let mut alloc_statements: Vec<_> = lowered.alloc_statements.into_iter().collect();
        alloc_statements.sort_by_key(|(k, _)| k.clone());
        let alloc_statements: Vec<_> = alloc_statements.into_iter().map(|(_, v)| v).collect();

        //let lowered = self.lower_node(&graph.root().lock().unwrap(), HashSet::new(), true);
        Program {
            rank: Statement::Function {
                ident: "rank".to_string(),
                args: vec![],
                body: Block {
                    statements: vec![Statement::Return {
                        value: Expr::Int(lowered.shape.len()),
                    }],
                },
            },
            shape: Statement::Function {
                ident: "shape".to_string(),
                args: self.input_args.clone(),
                body: Block {
                    statements: lowered
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(ind, (input_ind, dim))| Statement::Assignment {
                            left: Expr::Indexed {
                                ident: format!("shape"),
                                index: Box::new(Expr::Int(ind)),
                            },
                            right: Expr::Indexed {
                                ident: format!("d{input_ind}"),
                                index: Box::new(Expr::Int(*dim)),
                            },
                        })
                        .collect(),
                },
            },
            library: lowered.def_block,
            exec: Statement::Function {
                ident: "f".to_string(),
                args: self.input_args.clone(),
                body: Block {
                    statements: [alloc_statements, lowered.exec_block.statements].concat(),
                },
            },
        }
    }

    fn lower_node(
        &mut self,
        node: &Node,
        pruned_loops: HashSet<(char, usize)>,
        root: bool,
        memo: &mut HashMap<usize, Lowered>,
    ) -> Lowered {
        let lowered = match &node.body {
            NodeBody::Leaf => self.lower_leaf_node(&node.index),
            NodeBody::Interior {
                op,
                schedule,
                shape,
            } => match memo.get(&node.id) {
                Some(cached) => Lowered {
                    def_block: Block {
                        statements: Vec::new(),
                    },
                    exec_block: Block {
                        statements: Vec::new(),
                    },
                    ..cached.clone()
                },
                None => self.lower_interior_node(
                    &node.index,
                    &op,
                    &node.children(),
                    shape,
                    &schedule,
                    pruned_loops,
                    root,
                    memo,
                ),
            },
        };

        memo.insert(node.id, lowered.clone());

        lowered
    }

    fn lower_leaf_node(&mut self, index: &String) -> Lowered {
        let arg_ident = format!("in{}", self.input_array_counter);
        self.input_array_counter += 1;

        let char_indices = Self::get_char_indices(index);

        let loop_idents: HashMap<char, (String, String)> = char_indices
            .iter()
            .map(|char_index| {
                let bound_ident = format!("b{}", self.base_loop_counter);
                let iterator_ident = format!("i{}", self.base_loop_counter);
                self.base_loop_counter += 1;
                (*char_index, (bound_ident, iterator_ident))
            })
            .collect();

        // push array arg
        self.input_args.push(Arg {
            type_: Type::ArrayRef(false),
            ident: Expr::Ident(arg_ident.clone()),
        });

        // push dim args
        let dim_args = char_indices.iter().map(|c| Arg {
            type_: Type::Int(false),
            ident: Expr::Ident(loop_idents[c].0.clone()),
        });
        self.input_args.extend(dim_args.clone());

        Lowered {
            def_block: Block::default(),
            alloc_statements: HashMap::new(),
            exec_block: Block::default(),
            def_args: Vec::new(),
            loop_idents: loop_idents,
            store_ident: arg_ident,
            shape: index
                .chars()
                .enumerate()
                .map(|(ind, c)| (self.input_array_counter - 1, ind))
                .collect(),
        }
    }

    fn lower_interior_node(
        &mut self,
        index: &String,
        op: &char,
        children: &Vec<(Node, String)>,
        shape: &Vec<(usize, usize)>,
        schedule: &Schedule,
        pruned_loops: HashSet<(char, usize)>,
        root: bool,
        memo: &mut HashMap<usize, Lowered>,
    ) -> Lowered {
        let mut all_char_indices: Vec<char> = children
            .iter()
            .fold(HashSet::new(), |mut all_char_indices, (_child, index)| {
                all_char_indices.extend(index.chars());
                all_char_indices
            })
            .into_iter()
            .collect();
        all_char_indices.sort();

        let mut schedule = schedule.clone(); // Can we avoid this?
        if schedule.loop_order.is_empty() {
            schedule.loop_order = all_char_indices
                .iter()
                .map(|index| (index.clone(), 0))
                .collect();
        }
        schedule.compute_levels.resize(children.len(), 0);

        schedule.loop_order = schedule
            .loop_order
            .iter()
            .filter(|l| !pruned_loops.contains(l))
            .map(|l| *l)
            .collect();

        // recursively lower children
        // note: the reason this is a fold instead of a map is because the loop_idents are
        //       determined jointly with all siblings. those idents determined by the first child
        //       are passed to the lower call of the subsequent siblings.
        let (
            child_def_blocks,
            child_alloc_statements,
            mut child_exec_blocks, // mut so fragments can be pulled out for fusion
            mut child_def_args,
            loop_idents,
            child_store_idents,
            child_shapes,
        ): (
            Vec<Block>,
            HashMap<String, Statement>,
            Vec<Block>,
            Vec<Arg>,
            HashMap<char, (String, String)>,
            Vec<String>,
            Vec<Vec<(usize, usize)>>,
        ) = children.iter().enumerate().fold(
            (
                vec![],
                HashMap::new(),
                vec![],
                vec![],
                HashMap::new(),
                vec![],
                vec![],
            ),
            |(
                mut def_blocks,
                mut alloc_statements,
                mut exec_blocks,
                mut def_args,
                mut loop_idents,
                mut child_store_idents,
                mut child_shapes,
            ),
             (ind, (child, index))| {
                // for mapping between child indexing and current node indexing
                let child_to_current_index: HashMap<char, char> =
                    child.index.chars().zip(index.chars()).collect();
                let current_to_child_index: HashMap<char, char> =
                    index.chars().zip(child.index.chars()).collect();

                let pruned_loops: HashSet<(char, usize)> = schedule.loop_order
                    [..schedule.compute_levels[ind]]
                    .iter()
                    .map(|(c, rank)| (*current_to_child_index.get(&c).unwrap_or(&c), *rank))
                    .collect();

                let Lowered {
                    def_block: child_def_block,
                    alloc_statements: child_alloc_statements,
                    exec_block: child_exec_block,
                    def_args: child_def_args,
                    loop_idents: child_loop_idents,
                    store_ident: child_store_ident,
                    shape: child_shape,
                } = self.lower_node(&child, pruned_loops, false, memo);

                let child_loop_idents: HashMap<char, (String, String)> = child_loop_idents
                    .into_iter()
                    .map(|(c, x)| (*child_to_current_index.get(&c).unwrap_or(&c), x))
                    //.filter(|(c, _)| !loop_idents.contains_key(c))
                    .collect();

                def_blocks.push(child_def_block);
                alloc_statements.extend(child_alloc_statements);
                exec_blocks.push(child_exec_block);
                Self::merge_args(&mut def_args, child_def_args);
                loop_idents.extend(child_loop_idents);
                child_store_idents.push(child_store_ident);
                child_shapes.push(child_shape);

                (
                    def_blocks,
                    alloc_statements,
                    exec_blocks,
                    def_args,
                    loop_idents,
                    child_store_idents,
                    child_shapes,
                )
            },
        );

        let shape = shape
            .iter()
            .map(|(child_ind, dim)| child_shapes[*child_ind][*dim])
            .collect::<Vec<_>>();

        let store_ident = match root {
            true => "out".to_string(),
            false => {
                let ident = format!("s{}", self.store_counter);
                self.store_counter += 1;
                ident
            }
        };

        // create split factor idents
        let split_factor_idents: HashMap<char, Vec<String>> = schedule
            .splits
            .iter()
            .map(|(char_index, split_list)| {
                (
                    *char_index,
                    split_list
                        .iter()
                        .enumerate()
                        .map(|(ind, _split_factor)| {
                            let split_factor_ident = format!(
                                "{}_{ind}_{}",
                                loop_idents[char_index].0, self.split_factor_count
                            );
                            self.split_factor_count += 1;
                            split_factor_ident
                        })
                        .collect(),
                )
            })
            .collect();

        // create assignment statement for each split factor ident
        let split_factor_assignment_statements: Vec<Statement> = schedule
            .splits
            .iter()
            .flat_map(|(char_index, split_factors)| {
                split_factors
                    .iter()
                    .zip(split_factor_idents[char_index].iter())
                    .map(|(factor, ident)| Statement::Declaration {
                        ident: ident.clone(),
                        value: Expr::Int(*factor),
                        type_: Type::Int(false),
                    })
            })
            .collect();

        let mut alloc_statements = child_alloc_statements;
        if !root {
            alloc_statements.insert(
                store_ident.clone(),
                Statement::Declaration {
                    ident: store_ident.clone(),
                    value: Expr::Alloc {
                        initial_value: Box::new(Expr::Scalar(match op {
                            '>' => f32::NEG_INFINITY,
                            '<' => f32::INFINITY,
                            '*' => 1.,
                            _ => 0.,
                        })),
                        shape: index.chars().map(|c| loop_idents[&c].0.clone()).collect(),
                    },
                    type_: Type::Array(true),
                },
            );
        }

        // TODO: The mapping should probably be done in the present function instead of passing
        //       the hashmap here.
        // TODO: stop splitting ident map
        let op_statement = Self::create_op_statement(
            op,
            // bound_idents
            &loop_idents
                .iter()
                .map(|(c, (ident, _))| (*c, ident.clone()))
                .collect(),
            // base_iterator_idents
            &loop_idents
                .iter()
                .map(|(c, (_, ident))| (*c, ident.clone()))
                .collect(),
            &child_store_idents,
            &children
                .iter()
                .map(|(child, index)| index.clone())
                .collect(),
            &store_ident,
            &index,
        );

        // TODO: stop splitting ident map
        let mut loop_statements: Vec<Statement> = Self::create_empty_loop_statements(
            &schedule,
            &loop_idents
                .iter()
                .map(|(c, (_, ident))| (*c, ident.clone()))
                .collect(),
            &loop_idents
                .iter()
                .map(|(c, (ident, _))| (*c, ident.clone()))
                .collect(),
            &split_factor_idents,
            &index,
        );

        // partition fragments from blocks
        let (child_exec_fragments, remaining_blocks): (Vec<(usize, Block)>, Vec<(usize, Block)>) =
            child_exec_blocks
                .into_iter()
                .enumerate()
                .partition(|(i, _)| {
                    schedule
                        .compute_levels
                        .get(*i)
                        .map_or(false, |&level| level > 0)
                });

        // transform filtered items into desired format (level, block)
        let child_exec_fragments: Vec<(usize, Block)> = child_exec_fragments
            .into_iter()
            .map(|(i, block)| (schedule.compute_levels[i], block))
            .collect();

        // reassign remaining blocks back to child_exec_blocks
        child_exec_blocks = remaining_blocks
            .into_iter()
            .map(|(_, block)| block)
            .collect();

        // fuse any child kernel fragments into the appropriate loop bodies
        let n_loop_statements = loop_statements.len();
        for (ind, child_exec_fragment) in child_exec_fragments {
            let Statement::Loop { body, .. } = &mut loop_statements[n_loop_statements - ind] else {
                panic!("Expected `Statement` to be of `Loop` variant")
            };
            body.statements
                .extend(child_exec_fragment.statements.clone());
        }

        let initialization_loop_stack: Option<Statement> = (root && (*op == '>' || *op == '*'))
            .then(|| Statement::Loop {
                index: "i_".to_string(),
                bound: Expr::Op {
                    op: '*',
                    inputs: loop_idents
                        .iter()
                        .filter(|(c, _)| index.contains(**c))
                        .map(|(_, (bound_ident, _))| Expr::Ident(bound_ident.clone()))
                        .collect(),
                },
                body: Block {
                    statements: vec![Statement::Assignment {
                        left: Expr::Indexed {
                            ident: store_ident.clone(),
                            index: Box::new(Expr::Ident("i_".to_string())),
                        },
                        right: Expr::Scalar(match op {
                            '*' => 1.,
                            '>' => f32::NEG_INFINITY,
                            '<' => f32::INFINITY,
                            _ => panic!("Attempted to initialize non-reduction."),
                        }),
                    }],
                },
                parallel: true,
            });

        let loop_stack: Statement =
            loop_statements
                .into_iter()
                .fold(op_statement, |loop_stack, mut loop_| {
                    if let Statement::Loop { ref mut body, .. } = loop_ {
                        body.statements.push(loop_stack);
                    }
                    loop_
                });

        if root {
            // push array arg
            self.input_args.push(Arg {
                type_: Type::ArrayRef(true),
                ident: Expr::Ident(store_ident.clone()),
            });

            // push dim args
            let dim_args = (0..Self::get_char_indices(&index).len()).map(|ind| Arg {
                type_: Type::Int(false),
                ident: Expr::Ident(format!("{}_{ind}", store_ident.clone())),
            });
            self.input_args.extend(dim_args.clone());
        };

        let function_ident = format!("_{}", store_ident.clone());

        let exec_statements = [
            split_factor_assignment_statements,
            initialization_loop_stack.into_iter().collect(),
            vec![loop_stack],
        ]
        .concat();

        // this will get drained for full kernels and returned populated for fragments
        let mut def_args: Vec<Arg> = [
            child_store_idents
                .iter()
                .map(|ident| Arg {
                    type_: Type::ArrayRef(false),
                    ident: Expr::Ident(ident.clone()),
                })
                .collect::<Vec<_>>(),
            vec![Arg {
                type_: Type::ArrayRef(true),
                ident: Expr::Ident(store_ident.clone()),
            }],
            all_char_indices
                .iter()
                .map(|c| Arg {
                    type_: Type::Int(false),
                    ident: Expr::Ident(loop_idents[c].0.clone()),
                })
                .collect::<Vec<_>>(),
        ]
        .concat();

        Self::merge_args(&mut def_args, child_def_args);

        // this will get drained for full kernels and returned populated for fragments
        let call_args: Vec<Arg> = def_args
            .iter()
            .map(|arg| match (arg.type_.clone(), arg.ident.clone()) {
                (Type::ArrayRef(mutable), Expr::Ident(s)) => Arg {
                    type_: Type::ArrayRef(mutable),
                    ident: Expr::Ref(s, mutable),
                },
                (Type::Int(mutable), Expr::Ident(s)) => Arg {
                    type_: Type::ArrayRef(mutable),
                    ident: Expr::Ident(s),
                },
                _ => panic!("Invalid argument."),
            })
            .collect();

        let (def_block, exec_block) = if pruned_loops.is_empty() {
            let def_block = Block {
                statements: [
                    child_def_blocks
                        .into_iter()
                        .flat_map(|block| block.statements)
                        .collect(),
                    vec![Statement::Function {
                        ident: function_ident.clone(),
                        args: def_args.drain(..).collect(),
                        body: Block {
                            statements: exec_statements,
                        },
                    }],
                ]
                .concat(),
            };

            let call = Statement::Call {
                ident: function_ident.clone(),
                args: call_args,
            };

            let exec_block = Block {
                statements: [
                    child_exec_blocks
                        .into_iter()
                        .flat_map(|block| block.statements)
                        .collect(),
                    vec![call],
                ]
                .concat(),
            };

            (def_block, exec_block)
        } else {
            let exec_block = Block {
                statements: [
                    child_exec_blocks
                        .into_iter()
                        .flat_map(|block| block.statements)
                        .collect(),
                    exec_statements,
                ]
                .concat(),
            };
            (Block::default(), exec_block)
        };

        Lowered {
            def_block,
            alloc_statements,
            exec_block,
            def_args,
            loop_idents,
            store_ident,
            shape,
        }
    }

    fn create_op_statement(
        op: &char,
        bound_idents: &HashMap<char, String>,
        base_iterator_idents: &HashMap<char, String>,
        child_store_idents: &Vec<String>,
        child_indices: &Vec<String>,
        store_ident: &String,
        index: &String,
    ) -> Statement {
        assert_eq!(child_store_idents.len(), child_indices.len());

        let out_expr = Expr::Indexed {
            ident: store_ident.clone(),
            index: Box::new(Self::create_affine_index(
                index
                    .chars()
                    .map(|c| base_iterator_idents[&c].clone())
                    .collect(),
                index.chars().map(|c| bound_idents[&c].clone()).collect(),
            )),
        };

        let mut in_exprs: Vec<Expr> = child_store_idents
            .iter()
            .zip(child_indices.iter())
            .map(|(ident, index)| Expr::Indexed {
                ident: ident.clone(),
                index: Box::new(Self::create_affine_index(
                    index
                        .chars()
                        .map(|c| base_iterator_idents[&c].clone())
                        .collect(),
                    index.chars().map(|c| bound_idents[&c].clone()).collect(),
                )),
            })
            .collect();

        if in_exprs.len() == 1 && matches!(op, '+' | '*' | '>' | '<') {
            // Pushing to front here shouldn't be a problem unless we start allowing ops of
            // arbitrary inputs.
            in_exprs.insert(0, out_expr.clone());
            assert_eq!(
                in_exprs.len(),
                2,
                "Expected exactly two operands for op [{op}]."
            );
        }

        Statement::Assignment {
            left: out_expr,
            right: Expr::Op {
                op: *op,
                inputs: in_exprs,
            },
        }
    }

    fn create_empty_loop_statements(
        schedule: &Schedule,
        base_iterator_idents: &HashMap<char, String>,
        bound_idents: &HashMap<char, String>,
        split_factor_idents: &HashMap<char, Vec<String>>,
        index: &String,
    ) -> Vec<Statement> {
        let mut statements = vec![];

        let mut needs_index_reconstruction: HashSet<char> = schedule
            .splits
            .iter()
            .filter(|(_c, splits_factors)| splits_factors.len() > 0)
            .map(|(c, _splits_factors)| *c)
            .collect();

        let output_char_indices: HashSet<char> = index.chars().collect();
        for (char_index, rank) in schedule.loop_order.iter().rev() {
            let splits = schedule.splits.get(char_index);

            let index = if splits.is_some() && *rank > 0 {
                format!(
                    "{}_{}",
                    base_iterator_idents[&char_index].clone(),
                    (*rank - 1)
                )
            } else {
                base_iterator_idents[&char_index].clone()
            };

            let bound = match (splits, rank) {
                (None, _) => Expr::Ident(bound_idents[&char_index].clone()),
                (Some(_splits), 0) => Self::create_split_bound_expr(
                    &bound_idents[&char_index],
                    &split_factor_idents[&char_index],
                ),
                (Some(_splits), rank) => {
                    Expr::Ident(split_factor_idents[&char_index][*rank - 1].clone())
                }
            };

            statements.push(Statement::Loop {
                index: index.clone(),
                bound: bound,
                body: Block {
                    statements: if needs_index_reconstruction.remove(&char_index) {
                        Self::create_index_reconstruction_statements(
                            &base_iterator_idents[&char_index],
                            &bound_idents[&char_index],
                            &split_factor_idents[&char_index],
                        )
                    } else {
                        vec![]
                    },
                },
                parallel: output_char_indices.contains(&char_index),
            });
        }

        statements
    }

    fn create_split_bound_expr(
        base_bound_ident: &String,
        split_factors_idents: &Vec<String>,
    ) -> Expr {
        let tile_width_expr = Expr::Op {
            op: '*',
            inputs: split_factors_idents
                .iter()
                .map(|ident| Expr::Ident(ident.clone()))
                .collect(),
        };

        let numerator = Expr::Op {
            op: '-',
            inputs: vec![
                Expr::Op {
                    op: '+',
                    inputs: vec![
                        Expr::Ident(base_bound_ident.clone()),
                        tile_width_expr.clone(),
                    ],
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
        base_iterator_ident: &String,
        base_bound_ident: &String,
        split_factors_idents: &Vec<String>,
    ) -> Vec<Statement> {
        let factor_loop_widths: Vec<Expr> = split_factors_idents
            .iter()
            .map(|ident| Expr::Ident(ident.clone()))
            .collect();

        // number of elements per iteration of base loop
        let base_loop_tile_width = Expr::Op {
            op: '*',
            inputs: factor_loop_widths.clone(),
        };

        let mut widths = factor_loop_widths;
        widths.insert(0, base_loop_tile_width);

        let factor_loop_iterator: Vec<Expr> = (0..split_factors_idents.len())
            .map(|ind| Expr::Ident(format!("{}_{ind}", base_iterator_ident.clone())))
            .collect();

        let mut iterators = factor_loop_iterator;
        iterators.insert(0, Expr::Ident(base_iterator_ident.clone()));

        // highest rank iterator iterates elementwise; all others iterate tilewise;
        // remove prior to total_width calculation
        let ultimate_iterator = iterators
            .pop()
            .expect("Expected non-empty iterator list for index reconstruction");
        let _ = widths.pop();

        assert_eq!(widths.len(), iterators.len());
        let mut total_width: Vec<Expr> = widths
            .into_iter()
            .zip(iterators.into_iter())
            .map(|(width, iterator)| Expr::Op {
                op: '*',
                inputs: vec![width, iterator],
            })
            .collect();

        total_width.push(ultimate_iterator);

        let reconstructed_index = Expr::Op {
            op: '+',
            inputs: total_width,
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

    fn create_affine_index(indices: Vec<String>, bounds: Vec<String>) -> Expr {
        let d = indices.len();
        let mut sum_expr = None;
        for k in 0..d {
            let mut product_expr = None;
            for m in (k + 1)..d {
                product_expr = Some(match product_expr {
                    Some(expr) => Expr::Op {
                        op: '*',
                        inputs: vec![expr, Expr::Ident(bounds[m].clone())],
                    },
                    None => Expr::Ident(bounds[m].clone()),
                });
            }
            let partial_expr = match product_expr {
                Some(expr) => Expr::Op {
                    op: '*',
                    inputs: vec![Expr::Ident(indices[k].clone()), expr],
                },
                None => Expr::Ident(indices[k].clone()),
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

    /// Merge two arg lists without duplication, and preferring mutability
    /// Mostly written by ChatGPT o1
    fn merge_args(def_args: &mut Vec<Arg>, child_def_args: Vec<Arg>) {
        for arg in child_def_args {
            let ident = match &arg.ident {
                Expr::Ident(s) => s,
                _ => panic!("Invalid Arg.ident type."),
            };
            if let Some(existing) = def_args.iter_mut().find(|e| match &e.ident {
                Expr::Ident(ei) => ei == ident,
                _ => false,
            }) {
                let incoming_mutable = match &arg.type_ {
                    Type::Int(m) | Type::Array(m) | Type::ArrayRef(m) => *m,
                };
                if incoming_mutable {
                    match &mut existing.type_ {
                        Type::Int(em) | Type::Array(em) | Type::ArrayRef(em) => *em = true,
                    }
                }
            } else {
                def_args.push(arg);
            }
        }
    }
}

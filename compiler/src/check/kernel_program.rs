use std::collections::BTreeSet;
use std::fmt;

use crate::check::graph::validate_graph;
use crate::ir::common::DimRef;
use crate::ir::kernel_program::{
    Access, Action, Block, BufferId, Iter, Kernel, KernelProgram, LoopId, ScalarExpr,
};

pub fn validate_kernel_program(program: &KernelProgram) -> Result<(), ValidationError> {
    validate_buffers(program)?;
    validate_outputs(program)?;
    validate_kernel_graph(program)
}

fn validate_buffers(program: &KernelProgram) -> Result<(), ValidationError> {
    for (buffer_index, buffer) in program.buffers.iter().enumerate() {
        for dim in &buffer.shape.0 {
            validate_dim_ref(program, *dim)
                .map_err(|message| err(format!("buffer {} shape {}", buffer_index, message)))?;
        }
        for extent in &buffer.layout.0 {
            validate_dim_ref(program, extent.source)
                .map_err(|message| err(format!("buffer {} layout {}", buffer_index, message)))?;
        }
    }
    Ok(())
}

fn validate_outputs(program: &KernelProgram) -> Result<(), ValidationError> {
    for output in &program.outputs {
        validate_buffer_id(program, *output)
            .map_err(|message| err(format!("output {}", message)))?;
    }
    Ok(())
}

fn validate_kernel_graph(program: &KernelProgram) -> Result<(), ValidationError> {
    validate_graph(&program.graph, |kernel| {
        validate_kernel(program, kernel).map_err(|error| error.to_string())
    })
    .map_err(|error| err(error.to_string()))?;

    if program.graph.inputs.len() > program.buffers.len() {
        return Err(err(format!(
            "graph has {} inputs for {} buffers",
            program.graph.inputs.len(),
            program.buffers.len()
        )));
    }

    for (node_index, node) in program.graph.nodes.iter().enumerate() {
        if node.inputs.len() != node.inner.reads.len() {
            return Err(err(format!(
                "node {}: graph has {} inputs for {} kernel reads",
                node_index,
                node.inputs.len(),
                node.inner.reads.len()
            )));
        }
        if node.outputs.len() != node.inner.writes.len() {
            return Err(err(format!(
                "node {}: graph has {} outputs for {} kernel writes",
                node_index,
                node.outputs.len(),
                node.inner.writes.len()
            )));
        }
    }

    Ok(())
}

pub fn validate_kernel(program: &KernelProgram, kernel: &Kernel) -> Result<(), ValidationError> {
    let mut reads = BTreeSet::new();
    for buffer in &kernel.reads {
        validate_buffer_id(program, *buffer).map_err(|message| err(format!("read {}", message)))?;
        if !reads.insert(buffer.0) {
            return Err(err(format!("read repeats buffer {}", buffer.0)));
        }
    }

    let mut writes = BTreeSet::new();
    for buffer in &kernel.writes {
        validate_buffer_id(program, *buffer)
            .map_err(|message| err(format!("write {}", message)))?;
        if !writes.insert(buffer.0) {
            return Err(err(format!("write repeats buffer {}", buffer.0)));
        }
        if reads.contains(&buffer.0) {
            return Err(err(format!("buffer {} is both read and written", buffer.0)));
        }
    }

    let mut loops = BTreeSet::new();
    collect_loop_ids(&kernel.body, &mut loops)?;
    validate_block(program, kernel, &loops, &BTreeSet::new(), &kernel.body)
}

fn collect_loop_ids(block: &Block, loops: &mut BTreeSet<usize>) -> Result<(), ValidationError> {
    for action in &block.0 {
        if let Action::Loop { id, body, .. } = action {
            if !loops.insert(id.0) {
                return Err(err(format!("loop id {} is repeated", id.0)));
            }
            collect_loop_ids(body, loops)?;
        }
    }
    Ok(())
}

fn validate_block(
    program: &KernelProgram,
    kernel: &Kernel,
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    block: &Block,
) -> Result<(), ValidationError> {
    for action in &block.0 {
        match action {
            Action::Loop {
                id, extent, body, ..
            } => {
                validate_dim_ref(program, extent.source)
                    .map_err(|message| err(format!("loop {} extent {}", id.0, message)))?;
                validate_kernel_buffer(kernel, extent.source.buffer)
                    .map_err(|message| err(format!("loop {} extent {}", id.0, message)))?;
                let mut child_scope = scope.clone();
                child_scope.insert(id.0);
                validate_block(program, kernel, loops, &child_scope, body)?;
            }
            Action::Init {
                write, zero_checks, ..
            } => {
                validate_write_access(program, kernel, loops, scope, write)
                    .map_err(|message| err(format!("init {}", message)))?;
                for loop_id in zero_checks {
                    validate_loop_ref(loops, scope, *loop_id)
                        .map_err(|message| err(format!("init zero check {}", message)))?;
                }
            }
            Action::Compute { write, reads, .. } => {
                validate_write_access(program, kernel, loops, scope, write)
                    .map_err(|message| err(format!("compute write {}", message)))?;
                for (read_index, read) in reads.iter().enumerate() {
                    validate_access(program, kernel, loops, scope, read).map_err(|message| {
                        err(format!("compute read {} {}", read_index, message))
                    })?;
                }
            }
            Action::Snapshot { write, read } => {
                validate_write_access(program, kernel, loops, scope, write)
                    .map_err(|message| err(format!("snapshot write {}", message)))?;
                validate_access(program, kernel, loops, scope, read)
                    .map_err(|message| err(format!("snapshot read {}", message)))?;
            }
            Action::Scale { write, factor } => {
                validate_write_access(program, kernel, loops, scope, write)
                    .map_err(|message| err(format!("scale write {}", message)))?;
                validate_scale_expr(program, kernel, loops, scope, factor)
                    .map_err(|message| err(format!("scale factor {}", message)))?;
            }
        }
    }
    Ok(())
}

fn validate_scale_expr(
    program: &KernelProgram,
    kernel: &Kernel,
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    expr: &ScalarExpr,
) -> Result<(), String> {
    match expr {
        ScalarExpr::Access(access) => validate_access(program, kernel, loops, scope, access),
        ScalarExpr::Unary { arg, .. } => validate_scale_expr(program, kernel, loops, scope, arg),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            validate_scale_expr(program, kernel, loops, scope, lhs)?;
            validate_scale_expr(program, kernel, loops, scope, rhs)
        }
    }
}

fn validate_write_access(
    program: &KernelProgram,
    kernel: &Kernel,
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    access: &Access,
) -> Result<(), String> {
    validate_buffer_id(program, access.buffer)?;
    for iter in &access.index {
        validate_iter(loops, scope, iter)?;
    }
    if !kernel.writes.contains(&access.buffer) {
        return Err(format!(
            "writes buffer {} outside writeables",
            access.buffer.0
        ));
    }
    Ok(())
}

fn validate_access(
    program: &KernelProgram,
    kernel: &Kernel,
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    access: &Access,
) -> Result<(), String> {
    validate_kernel_buffer(kernel, access.buffer)?;
    validate_buffer_id(program, access.buffer)?;
    for iter in &access.index {
        validate_iter(loops, scope, iter)?;
    }
    Ok(())
}

fn validate_iter(
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    iter: &Iter,
) -> Result<(), String> {
    match iter {
        Iter::Raw(loop_id) => validate_loop_ref(loops, scope, *loop_id),
        Iter::Reconstructed {
            loops: ids,
            factors,
        } => {
            if ids.is_empty() {
                return Err("reconstructed iterator has no loops".to_string());
            }
            if factors.len().saturating_add(1) != ids.len() {
                return Err(format!(
                    "reconstructed iterator has {} loops and {} factors",
                    ids.len(),
                    factors.len()
                ));
            }
            for loop_id in ids {
                validate_loop_ref(loops, scope, *loop_id)?;
            }
            Ok(())
        }
    }
}

fn validate_loop_ref(
    loops: &BTreeSet<usize>,
    scope: &BTreeSet<usize>,
    loop_id: LoopId,
) -> Result<(), String> {
    if !loops.contains(&loop_id.0) {
        return Err(format!("references nonexistent loop {}", loop_id.0));
    }
    if !scope.contains(&loop_id.0) {
        return Err(format!("references loop {} outside scope", loop_id.0));
    }
    Ok(())
}

fn validate_kernel_buffer(kernel: &Kernel, buffer: BufferId) -> Result<(), String> {
    if !kernel.reads.contains(&buffer) && !kernel.writes.contains(&buffer) {
        return Err(format!("references buffer {} outside params", buffer.0));
    }
    Ok(())
}

fn validate_dim_ref(program: &KernelProgram, dim_ref: DimRef<BufferId>) -> Result<(), String> {
    validate_buffer_id(program, dim_ref.buffer)?;
    let shape_rank = program.buffers[dim_ref.buffer.0].shape.0.len();
    if dim_ref.dim >= shape_rank {
        return Err(format!(
            "references nonexistent dim {} of buffer {}",
            dim_ref.dim, dim_ref.buffer.0
        ));
    }
    Ok(())
}

fn validate_buffer_id(program: &KernelProgram, buffer: BufferId) -> Result<(), String> {
    if buffer.0 >= program.buffers.len() {
        return Err(format!("references nonexistent buffer {}", buffer.0));
    }
    Ok(())
}

fn err(message: impl Into<String>) -> ValidationError {
    ValidationError {
        message: message.into(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::validate_kernel_program;
    use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
    use crate::ir::graph::{Graph, Node, Output};
    use crate::ir::kernel_program::{
        Access, Action, Block, Buffer, BufferId, BufferKind, BufferLayout, BufferShape, Kernel,
        KernelProgram, LoopId, TailGuard,
    };

    fn program() -> KernelProgram {
        KernelProgram {
            buffers: vec![
                Buffer {
                    kind: BufferKind::Input,
                    shape: BufferShape(vec![DimRef {
                        buffer: BufferId(0),
                        dim: 0,
                    }]),
                    layout: BufferLayout(vec![Extent {
                        source: DimRef {
                            buffer: BufferId(0),
                            dim: 0,
                        },
                        kind: ExtentKind::Semantic,
                    }]),
                },
                Buffer {
                    kind: BufferKind::Output,
                    shape: BufferShape(vec![DimRef {
                        buffer: BufferId(0),
                        dim: 0,
                    }]),
                    layout: BufferLayout(vec![Extent {
                        source: DimRef {
                            buffer: BufferId(1),
                            dim: 0,
                        },
                        kind: ExtentKind::Semantic,
                    }]),
                },
            ],
            outputs: vec![BufferId(1)],
            graph: Graph {
                inputs: vec![crate::ir::graph::Input],
                nodes: vec![Node {
                    inner: Kernel {
                        reads: vec![BufferId(0)],
                        writes: vec![BufferId(1)],
                        body: Block(vec![Action::Loop {
                            id: LoopId(0),
                            extent: Extent {
                                source: DimRef {
                                    buffer: BufferId(1),
                                    dim: 0,
                                },
                                kind: ExtentKind::Semantic,
                            },
                            guard: TailGuard(false),
                            body: Block(vec![Action::Compute {
                                op: Op::Add,
                                write: Access {
                                    buffer: BufferId(1),
                                    index: vec![crate::ir::kernel_program::Iter::Raw(LoopId(0))],
                                },
                                reads: vec![Access {
                                    buffer: BufferId(0),
                                    index: vec![crate::ir::kernel_program::Iter::Raw(LoopId(0))],
                                }],
                            }]),
                        }]),
                    },
                    inputs: vec![crate::ir::graph::Source::Input(crate::ir::graph::InputId(
                        0,
                    ))],
                    outputs: vec![Output],
                }],
                outputs: vec![crate::ir::graph::Source::Node(
                    crate::ir::graph::NodeId(0),
                    crate::ir::graph::OutputId(0),
                )],
            },
        }
    }

    #[test]
    fn accepts_kernel_program() {
        assert!(validate_kernel_program(&program()).is_ok());
    }

    #[test]
    fn rejects_invalid_output_buffer() {
        let mut program = program();
        program.outputs = vec![BufferId(2)];

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(error.to_string(), "output references nonexistent buffer 2");
    }

    #[test]
    fn rejects_write_outside_writeables() {
        let mut program = program();
        program.graph.nodes[0].inner.writes = vec![];
        program.graph.nodes[0].outputs = vec![];
        if let Action::Loop { extent, .. } = &mut program.graph.nodes[0].inner.body.0[0] {
            extent.source = DimRef {
                buffer: BufferId(0),
                dim: 0,
            };
        }

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: compute write writes buffer 1 outside writeables"
        );
    }

    #[test]
    fn rejects_loop_outside_scope() {
        let mut program = program();
        let action = match &mut program.graph.nodes[0].inner.body.0[0] {
            Action::Loop { body, .. } => &mut body.0[0],
            _ => unreachable!(),
        };
        if let Action::Compute { write, .. } = action {
            write.index = vec![crate::ir::kernel_program::Iter::Raw(LoopId(1))];
        }

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: compute write references nonexistent loop 1"
        );
    }

    #[test]
    fn rejects_duplicate_loop_id() {
        let mut program = program();
        let Action::Loop { body, extent, .. } = &mut program.graph.nodes[0].inner.body.0[0] else {
            unreachable!();
        };
        body.0.insert(
            0,
            Action::Loop {
                id: LoopId(0),
                extent: extent.clone(),
                guard: TailGuard(false),
                body: Block(vec![]),
            },
        );

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(error.to_string(), "node 0: loop id 0 is repeated");
    }

    #[test]
    fn rejects_invalid_layout_dim_ref() {
        let mut program = program();
        program.buffers[1].layout.0[0].source = DimRef {
            buffer: BufferId(1),
            dim: 1,
        };

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "buffer 1 layout references nonexistent dim 1 of buffer 1"
        );
    }

    #[test]
    fn rejects_kernel_graph_read_arity_mismatch() {
        let mut program = program();
        program.graph.nodes[0].inputs = vec![];

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: graph has 0 inputs for 1 kernel reads"
        );
    }

    #[test]
    fn rejects_read_access_outside_params() {
        let mut program = program();
        program.buffers.push(Buffer {
            kind: BufferKind::Intermediate,
            shape: BufferShape(vec![DimRef {
                buffer: BufferId(0),
                dim: 0,
            }]),
            layout: BufferLayout(vec![Extent {
                source: DimRef {
                    buffer: BufferId(2),
                    dim: 0,
                },
                kind: ExtentKind::Semantic,
            }]),
        });
        let Action::Loop { body, .. } = &mut program.graph.nodes[0].inner.body.0[0] else {
            unreachable!();
        };
        let Action::Compute { reads, .. } = &mut body.0[0] else {
            unreachable!();
        };
        reads[0].buffer = BufferId(2);

        let error = validate_kernel_program(&program).unwrap_err();
        assert_eq!(
            error.to_string(),
            "node 0: compute read 0 references buffer 2 outside params"
        );
    }
}

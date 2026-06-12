use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::check::parallel_module::validate_parallel_module;
use crate::ir::common::{DimRef, Extent, ExtentKind, Op};
use crate::ir::exec_plan::{
    Arg, BufferRef, ExecutionDim, Input, Intermediate, KernelRef, Local, Param,
};
use crate::ir::kernel_program::{Access, Action, Block, BufferScope, Iter, LoopId, ScalarExpr};
use crate::ir::parallel_module::{DeviceKernel, DeviceKernelId, HostStep, ParallelModule};

pub fn render(module: &ParallelModule) -> Result<String, RenderError> {
    validate_parallel_module(module).map_err(RenderError::from_parallel_module)?;
    Builder { module }.render()
}

struct Builder<'a> {
    module: &'a ParallelModule,
}

impl<'a> Builder<'a> {
    fn render(&self) -> Result<String, RenderError> {
        let kernels = self
            .module
            .kernels
            .iter()
            .enumerate()
            .map(|(index, kernel)| self.render_kernel(index, kernel))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n\n");

        Ok(format!(
            r#"#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {{ const float* data; const size_t* shape; const size_t rank; }} Tensor;
typedef struct {{ float* data; const size_t* shape; const size_t rank; }} TensorMut;
typedef struct {{ const float* data; const size_t* shape; const size_t* layout; }} View;
typedef struct {{ float* data; const size_t* shape; const size_t* layout; }} ViewMut;

#define CUDA_CHECK(expr) do {{ cudaError_t err__ = (expr); if (err__ != cudaSuccess) abort(); }} while (0)

static size_t count_shape(size_t ndims, const size_t* shape) {{
  size_t n = 1;
  for (size_t i = 0; i < ndims; ++i) n *= shape[i];
  return n;
}}

static size_t* copy_size_array(size_t n, const size_t* src) {{
  if (n == 0) return NULL;
  size_t* dst = NULL;
  CUDA_CHECK(cudaMallocManaged((void**)&dst, n * sizeof(size_t)));
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
  return dst;
}}

static View copy_input_to_device(const Tensor* t) {{
  size_t n = count_shape(t->rank, t->shape);
  float* data = NULL;
  CUDA_CHECK(cudaMalloc((void**)&data, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(data, t->data, n * sizeof(float), cudaMemcpyHostToDevice));
  size_t* shape = copy_size_array(t->rank, t->shape);
  return (View){{ .data = data, .shape = shape, .layout = shape }};
}}

static ViewMut copy_output_to_device(const TensorMut* t) {{
  size_t n = count_shape(t->rank, t->shape);
  float* data = NULL;
  CUDA_CHECK(cudaMalloc((void**)&data, n * sizeof(float)));
  size_t* shape = copy_size_array(t->rank, t->shape);
  return (ViewMut){{ .data = data, .shape = shape, .layout = shape }};
}}

static void copy_output_to_host(const ViewMut* src, TensorMut* dst) {{
  size_t n = count_shape(dst->rank, dst->shape);
  CUDA_CHECK(cudaMemcpy(dst->data, src->data, n * sizeof(float), cudaMemcpyDeviceToHost));
}}

static ViewMut alloc_view_mut(size_t ndims, const size_t* layout, const size_t* shape) {{
  size_t n = count_shape(ndims, layout);
  float* data = NULL;
  CUDA_CHECK(cudaMalloc((void**)&data, n * sizeof(float)));
  return (ViewMut){{ .data = data, .shape = copy_size_array(ndims, shape), .layout = copy_size_array(ndims, layout) }};
}}

static void free_view(View view) {{
  CUDA_CHECK(cudaFree((void*)view.data));
  CUDA_CHECK(cudaFree((void*)view.shape));
}}

static void free_view_mut(ViewMut view) {{
  CUDA_CHECK(cudaFree(view.data));
  CUDA_CHECK(cudaFree((void*)view.shape));
  if (view.layout != view.shape) CUDA_CHECK(cudaFree((void*)view.layout));
}}

static View view_mut_as_view(const ViewMut* t) {{
  return (View){{ .data = t->data, .shape = t->shape, .layout = t->layout }};
}}

{count}

{ranks}

{shapes}

{kernels}

{exec}
"#,
            count = self.render_count(),
            ranks = self.render_ranks(),
            shapes = self.render_shapes(),
            kernels = kernels,
            exec = self.render_exec()?,
        ))
    }

    fn render_count(&self) -> String {
        format!(
            "extern \"C\" size_t count(void) {{\n  return {};\n}}",
            self.module.count
        )
    }

    fn render_ranks(&self) -> String {
        let mut lines = Vec::new();
        for (output, rank) in self.module.ranks.iter().enumerate() {
            lines.push(format!("ranks[{output}] = {rank};"));
        }
        lines.push("return;".to_string());
        format!(
            "extern \"C\" void ranks(size_t* ranks) {{\n{}\n}}",
            indent_lines(lines)
        )
    }

    fn render_shapes(&self) -> String {
        let mut lines = Vec::new();
        for (output, shape) in self.module.shapes.iter().enumerate() {
            for (dim, source) in shape.0.iter().enumerate() {
                lines.push(format!(
                    "shapes[{output}][{dim}] = inputs[{}].shape[{}];",
                    source.buffer.0, source.dim
                ));
            }
        }
        lines.push("return;".to_string());
        format!(
            "extern \"C\" void shapes(const Tensor* inputs, size_t** shapes) {{\n{}\n}}",
            indent_lines(lines)
        )
    }

    fn render_exec(&self) -> Result<String, RenderError> {
        let mut lines = Vec::new();
        for input in 0..self.module.buffers.inputs.len() {
            lines.push(format!(
                "View {} = copy_input_to_device(&inputs[{input}]);",
                input_ident(Input(input))
            ));
        }
        for output in 0..self.module.buffers.outputs.len() {
            lines.push(format!(
                "ViewMut {} = copy_output_to_device(&outputs[{output}]);",
                output_ident(crate::ir::exec_plan::Output(output))
            ));
        }

        for step in &self.module.exec.0 {
            match step {
                HostStep::Alloc(intermediate) => {
                    let buffer = &self.module.buffers.intermediates[intermediate.0];
                    let ident = intermediate_ident(*intermediate);
                    lines.push(render_array_decl(
                        &format!("{ident}_layout"),
                        buffer
                            .layout
                            .0
                            .iter()
                            .map(|extent| self.input_extent(extent))
                            .collect(),
                    ));
                    lines.push(render_array_decl(
                        &format!("{ident}_shape"),
                        buffer
                            .shape
                            .0
                            .iter()
                            .copied()
                            .map(|dim| self.input_dim(dim))
                            .collect(),
                    ));
                    lines.push(format!(
                        "ViewMut {ident} = alloc_view_mut({}, {}, {});",
                        buffer.layout.0.len(),
                        array_arg(&format!("{ident}_layout"), buffer.layout.0.len()),
                        array_arg(&format!("{ident}_shape"), buffer.shape.0.len())
                    ));
                }
                HostStep::Launch {
                    kernel,
                    reads,
                    writes,
                } => {
                    lines.extend(self.render_launch(*kernel, reads, writes)?);
                }
                HostStep::Free(intermediate) => {
                    lines.push(format!(
                        "free_view_mut({});",
                        intermediate_ident(*intermediate)
                    ));
                }
            }
        }

        for output in 0..self.module.buffers.outputs.len() {
            lines.push(format!(
                "copy_output_to_host(&{}, &outputs[{output}]);",
                output_ident(crate::ir::exec_plan::Output(output))
            ));
        }
        for output in 0..self.module.buffers.outputs.len() {
            lines.push(format!(
                "free_view_mut({});",
                output_ident(crate::ir::exec_plan::Output(output))
            ));
        }
        for input in 0..self.module.buffers.inputs.len() {
            lines.push(format!("free_view({});", input_ident(Input(input))));
        }
        lines.push("return;".to_string());

        Ok(format!(
            "extern \"C\" void exec(const Tensor* inputs, TensorMut* outputs) {{\n{}\n}}",
            indent_lines(lines)
        ))
    }

    fn render_launch(
        &self,
        kernel_id: DeviceKernelId,
        reads: &[BufferRef],
        writes: &[BufferRef],
    ) -> Result<Vec<String>, RenderError> {
        let kernel =
            self.module.kernels.get(kernel_id.0).ok_or_else(|| {
                RenderError::new(format!("unknown device kernel {}", kernel_id.0))
            })?;
        let semantic_dims = self.kernel_semantic_dims(kernel_id.0, kernel)?;
        let mut lines = Vec::new();
        let read_array = format!("f{}_reads", kernel_id.0);
        let write_array = format!("f{}_writes", kernel_id.0);
        lines.push(format!("View* {read_array} = NULL;"));
        lines.push(format!("ViewMut* {write_array} = NULL;"));
        lines.push(format!(
            "CUDA_CHECK(cudaMallocManaged((void**)&{read_array}, {} * sizeof(View)));",
            reads.len()
        ));
        lines.push(format!(
            "CUDA_CHECK(cudaMallocManaged((void**)&{write_array}, {} * sizeof(ViewMut)));",
            writes.len()
        ));
        for (index, buffer) in reads.iter().enumerate() {
            lines.push(format!(
                "{read_array}[{index}] = {};",
                self.buffer_read_expr(*buffer)?
            ));
        }
        for (index, buffer) in writes.iter().enumerate() {
            lines.push(format!(
                "{write_array}[{index}] = {};",
                self.buffer_write_expr(*buffer)?
            ));
        }

        let dims = LaunchDims {
            groups: kernel
                .execution
                .groups
                .iter()
                .map(|dim| self.host_execution_extent(dim, reads, writes, &semantic_dims))
                .collect::<Result<Vec<_>, _>>()?,
            lanes: kernel
                .execution
                .lanes
                .iter()
                .map(|dim| self.host_execution_extent(dim, reads, writes, &semantic_dims))
                .collect::<Result<Vec<_>, _>>()?,
        };
        lines.push(format!(
            "dim3 f{}_grid{};",
            kernel_id.0,
            dims.dim3(&dims.groups)
        ));
        lines.push(format!(
            "dim3 f{}_block{};",
            kernel_id.0,
            dims.dim3(&dims.lanes)
        ));
        if !dims.lanes.is_empty() {
            lines.push(format!(
                "if ({}) abort();",
                dims.lanes.join(" * ") + " > 1024"
            ));
        }
        lines.push(format!(
            "f{}<<<f{}_grid, f{}_block>>>({read_array}, {write_array});",
            kernel_id.0, kernel_id.0, kernel_id.0
        ));
        lines.push("CUDA_CHECK(cudaGetLastError());".to_string());
        lines.push("CUDA_CHECK(cudaDeviceSynchronize());".to_string());
        lines.push(format!("CUDA_CHECK(cudaFree({read_array}));"));
        lines.push(format!("CUDA_CHECK(cudaFree({write_array}));"));
        Ok(lines)
    }

    fn render_kernel(
        &self,
        kernel_index: usize,
        kernel: &DeviceKernel,
    ) -> Result<String, RenderError> {
        let mut lower = KernelRenderer {
            initialized: initialized_buffers(&kernel.body),
            loops: BTreeMap::new(),
            loop_stack: Vec::new(),
            semantic_dims: self.kernel_semantic_dims(kernel_index, kernel)?,
        };
        let mut lines = Vec::new();
        lines.extend(lower.bind_execution(kernel));
        lines.extend(lower.render_locals(kernel)?);
        lines.extend(lower.render_execution_guards(kernel)?);
        lines.extend(lower.render_block(&kernel.body)?);
        lines.extend(lower.close_execution_guards(kernel));
        Ok(format!(
            "__global__ void f{}(const View* readonlys, ViewMut* writeables) {{\n{}\n}}",
            kernel_index,
            indent_lines(lines)
        ))
    }

    fn kernel_semantic_dims(
        &self,
        kernel_index: usize,
        kernel: &DeviceKernel,
    ) -> Result<Vec<(DimRef<KernelRef>, DimRef<Input>)>, RenderError> {
        let launch = self
            .module
            .exec
            .0
            .iter()
            .find_map(|step| match step {
                HostStep::Launch {
                    kernel,
                    reads,
                    writes,
                } if kernel.0 == kernel_index => Some((reads, writes)),
                _ => None,
            })
            .ok_or_else(|| {
                RenderError::new(format!("kernel {} is never launched", kernel_index))
            })?;
        let mut dims = Vec::new();
        for (ind, buffer) in launch.0.iter().copied().enumerate() {
            self.extend_param_semantic_dims(&mut dims, Arg::Readonly, ind, buffer)?;
        }
        for (ind, buffer) in launch.1.iter().copied().enumerate() {
            self.extend_param_semantic_dims(&mut dims, Arg::Writeable, ind, buffer)?;
        }
        for (local, buffer) in kernel.locals.iter().enumerate() {
            for (dim, source) in buffer.buffer.shape.0.iter().copied().enumerate() {
                dims.push((
                    DimRef {
                        buffer: KernelRef::Local(Local(local)),
                        dim,
                    },
                    source,
                ));
            }
            for (dim, extent) in buffer.buffer.layout.0.iter().enumerate() {
                dims.push((
                    DimRef {
                        buffer: KernelRef::Local(Local(local)),
                        dim,
                    },
                    extent.source,
                ));
            }
        }
        Ok(dims)
    }

    fn extend_param_semantic_dims(
        &self,
        dims: &mut Vec<(DimRef<KernelRef>, DimRef<Input>)>,
        arg: Arg,
        ind: usize,
        buffer: BufferRef,
    ) -> Result<(), RenderError> {
        for (dim, source) in self.buffer_shape(buffer)?.0.iter().copied().enumerate() {
            dims.push((
                DimRef {
                    buffer: KernelRef::Param(Param { arg, ind }),
                    dim,
                },
                source,
            ));
        }
        Ok(())
    }

    fn buffer_shape(&self, buffer: BufferRef) -> Result<&crate::ir::exec_plan::Shape, RenderError> {
        match buffer {
            BufferRef::Input(input) => self
                .module
                .buffers
                .inputs
                .get(input.0)
                .map(|buffer| &buffer.shape),
            BufferRef::Intermediate(intermediate) => self
                .module
                .buffers
                .intermediates
                .get(intermediate.0)
                .map(|buffer| &buffer.shape),
            BufferRef::Output(output) => self
                .module
                .buffers
                .outputs
                .get(output.0)
                .map(|buffer| &buffer.shape),
        }
        .ok_or_else(|| RenderError::new("launch references nonexistent buffer"))
    }

    fn input_dim(&self, dim: DimRef<Input>) -> String {
        format!("inputs[{}].shape[{}]", dim.buffer.0, dim.dim)
    }

    fn input_extent(&self, extent: &Extent<Input>) -> String {
        extent_expr(self.input_dim(extent.source), &extent.kind)
    }

    fn buffer_read_expr(&self, buffer: BufferRef) -> Result<String, RenderError> {
        Ok(match buffer {
            BufferRef::Input(input) => input_ident(input),
            BufferRef::Intermediate(intermediate) => {
                format!("view_mut_as_view(&{})", intermediate_ident(intermediate))
            }
            BufferRef::Output(output) => format!("view_mut_as_view(&{})", output_ident(output)),
        })
    }

    fn buffer_write_expr(&self, buffer: BufferRef) -> Result<String, RenderError> {
        Ok(match buffer {
            BufferRef::Input(_) => {
                return Err(RenderError::new("launch writes to input buffer"));
            }
            BufferRef::Intermediate(intermediate) => intermediate_ident(intermediate),
            BufferRef::Output(output) => output_ident(output),
        })
    }

    fn host_execution_extent(
        &self,
        dim: &ExecutionDim,
        reads: &[BufferRef],
        writes: &[BufferRef],
        semantic_dims: &[(DimRef<KernelRef>, DimRef<Input>)],
    ) -> Result<String, RenderError> {
        if let Some((_, source)) = semantic_dims
            .iter()
            .find(|(kernel_dim, _)| *kernel_dim == dim.extent.source)
        {
            return Ok(extent_expr(self.input_dim(*source), &dim.extent.kind));
        }

        let source = match dim.extent.source.buffer {
            KernelRef::Param(param) => match param.arg {
                Arg::Readonly => self.buffer_read_expr(
                    *reads
                        .get(param.ind)
                        .ok_or_else(|| RenderError::new("execution references missing read"))?,
                )?,
                Arg::Writeable => self.buffer_write_expr(
                    *writes
                        .get(param.ind)
                        .ok_or_else(|| RenderError::new("execution references missing write"))?,
                )?,
            },
            KernelRef::Local(_) => {
                return Err(RenderError::new(
                    "execution extent cannot be sourced from a local buffer",
                ));
            }
        };
        Ok(extent_expr(
            format!("{source}.shape[{}]", dim.extent.source.dim),
            &dim.extent.kind,
        ))
    }
}

struct LaunchDims {
    groups: Vec<String>,
    lanes: Vec<String>,
}

impl LaunchDims {
    fn dim3(&self, dims: &[String]) -> String {
        let x = dims.get(0).cloned().unwrap_or_else(|| "1".to_string());
        let y = dims.get(1).cloned().unwrap_or_else(|| "1".to_string());
        let z = dims.get(2).cloned().unwrap_or_else(|| "1".to_string());
        format!("({x}, {y}, {z})")
    }
}

struct KernelRenderer {
    initialized: BTreeSet<KernelRef>,
    loops: BTreeMap<LoopId, LoopInfo>,
    loop_stack: Vec<LoopId>,
    semantic_dims: Vec<(DimRef<KernelRef>, DimRef<Input>)>,
}

impl KernelRenderer {
    fn bind_execution(&mut self, kernel: &DeviceKernel) -> Vec<String> {
        let mut lines = Vec::new();
        for (index, dim) in kernel.execution.groups.iter().enumerate() {
            let expr = ["blockIdx.x", "blockIdx.y", "blockIdx.z"][index].to_string();
            self.push_loop_with_expr(
                dim.id,
                expr.clone(),
                dim.extent.source,
                dim.extent.kind.clone(),
            );
            lines.push(format!("const size_t {} = {};", loop_ident(dim.id), expr));
        }
        for (index, dim) in kernel.execution.lanes.iter().enumerate() {
            let expr = ["threadIdx.x", "threadIdx.y", "threadIdx.z"][index].to_string();
            self.push_loop_with_expr(
                dim.id,
                expr.clone(),
                dim.extent.source,
                dim.extent.kind.clone(),
            );
            lines.push(format!("const size_t {} = {};", loop_ident(dim.id), expr));
        }
        lines
    }

    fn render_locals(&self, kernel: &DeviceKernel) -> Result<Vec<String>, RenderError> {
        let mut lines = Vec::new();
        for (local, buffer) in kernel.locals.iter().enumerate() {
            if buffer.scope == BufferScope::Global {
                return Err(RenderError::new(format!("local {} is global", local)));
            }
            let ident = local_ident(Local(local));
            let layout = buffer
                .buffer
                .layout
                .0
                .iter()
                .map(|extent| Ok(extent_expr(self.input_dim(extent.source)?, &extent.kind)))
                .collect::<Result<Vec<_>, _>>()?;
            let shape = buffer
                .buffer
                .shape
                .0
                .iter()
                .copied()
                .map(|dim| self.input_dim(dim))
                .collect::<Result<Vec<_>, _>>()?;
            let size = static_layout_size(&layout)
                .ok_or_else(|| RenderError::new(format!("local {} layout is not static", local)))?;
            lines.push(format!("float {ident}_data[{size}];"));
            lines.push(render_array_decl(&format!("{ident}_layout"), layout));
            lines.push(render_array_decl(&format!("{ident}_shape"), shape));
            lines.push(format!(
                "ViewMut {ident} = (ViewMut){{ .data = {ident}_data, .shape = {}, .layout = {} }};",
                array_arg(&format!("{ident}_shape"), buffer.buffer.shape.0.len()),
                array_arg(&format!("{ident}_layout"), buffer.buffer.layout.0.len())
            ));
        }
        Ok(lines)
    }

    fn render_execution_guards(&self, kernel: &DeviceKernel) -> Result<Vec<String>, RenderError> {
        let mut lines = Vec::new();
        for dim in kernel
            .execution
            .groups
            .iter()
            .chain(kernel.execution.lanes.iter())
        {
            if dim.guard.0 {
                lines.push(format!(
                    "if ({} < {}) {{",
                    self.reconstruct_extent_index(dim.extent.source)?,
                    self.buffer_dim(dim.extent.source)
                ));
            }
        }
        Ok(lines)
    }

    fn close_execution_guards(&self, kernel: &DeviceKernel) -> Vec<String> {
        kernel
            .execution
            .groups
            .iter()
            .chain(kernel.execution.lanes.iter())
            .filter(|dim| dim.guard.0)
            .map(|_| "}".to_string())
            .collect()
    }

    fn render_block(&mut self, block: &Block<KernelRef>) -> Result<Vec<String>, RenderError> {
        let mut lines = Vec::new();
        for action in &block.0 {
            lines.extend(self.render_action(action)?);
        }
        Ok(lines)
    }

    fn render_action(&mut self, action: &Action<KernelRef>) -> Result<Vec<String>, RenderError> {
        Ok(match action {
            Action::Loop {
                id,
                extent,
                guard,
                body,
                ..
            } => {
                let iter = loop_ident(*id);
                self.push_loop_with_expr(*id, iter.clone(), extent.source, extent.kind.clone());
                let mut lines = vec![format!(
                    "for (size_t {iter} = 0; {iter} < {}; ++{iter}) {{",
                    extent_expr(self.buffer_dim(extent.source), &extent.kind)
                )];
                if guard.0 {
                    lines.push(format!(
                        "if ({} < {}) {{",
                        self.reconstruct_extent_index(extent.source)?,
                        self.buffer_dim(extent.source)
                    ));
                }
                lines.extend(
                    self.render_block(body)?
                        .into_iter()
                        .map(|line| format!("  {line}")),
                );
                if guard.0 {
                    lines.push("}".to_string());
                }
                lines.push("}".to_string());
                self.pop_loop(*id);
                lines
            }
            Action::Init {
                op,
                write,
                zero_checks,
            } => {
                let set = format!("{} = {};", self.lower_write(write)?, init_value(*op)?);
                if let Some(cond) = self.zero_check_condition(zero_checks)? {
                    vec![
                        format!("if ({cond}) {{"),
                        format!("  {set}"),
                        "}".to_string(),
                    ]
                } else {
                    vec![set]
                }
            }
            Action::Compute { op, write, reads } => {
                let dst = self.lower_write(write)?;
                let mut args = Vec::new();
                if self.initialized.contains(&write.buffer) {
                    args.push(dst.clone());
                }
                for read in reads {
                    args.push(self.lower_read(read)?);
                }
                vec![format!("{} = {};", dst, render_op(*op, &args)?)]
            }
            Action::Snapshot { write, read } => {
                vec![format!(
                    "{} = {};",
                    self.lower_write(write)?,
                    self.lower_read(read)?
                )]
            }
            Action::Scale { write, factor } => {
                let dst = self.lower_write(write)?;
                vec![format!(
                    "{dst} = {dst} * {};",
                    self.lower_scale_expr(factor)?
                )]
            }
        })
    }

    fn lower_scale_expr(&self, expr: &ScalarExpr<KernelRef>) -> Result<String, RenderError> {
        match expr {
            ScalarExpr::Access(access) => self.lower_read(access),
            ScalarExpr::Unary { op, arg } => render_op(*op, &[self.lower_scale_expr(arg)?]),
            ScalarExpr::Binary { op, lhs, rhs } => render_op(
                *op,
                &[self.lower_scale_expr(lhs)?, self.lower_scale_expr(rhs)?],
            ),
        }
    }

    fn lower_write(&self, access: &Access<KernelRef>) -> Result<String, RenderError> {
        Ok(format!(
            "{}.data[{}]",
            self.buffer_expr(access.buffer),
            self.flat_index(access.buffer, &access.index)?
        ))
    }

    fn lower_read(&self, access: &Access<KernelRef>) -> Result<String, RenderError> {
        Ok(format!(
            "{}.data[{}]",
            self.buffer_expr(access.buffer),
            self.flat_index(access.buffer, &access.index)?
        ))
    }

    fn flat_index(&self, buffer: KernelRef, index: &[Iter]) -> Result<String, RenderError> {
        if index.is_empty() {
            return Ok("0".to_string());
        }
        let dims = index
            .iter()
            .map(|iter| self.lower_iter(iter))
            .collect::<Result<Vec<_>, _>>()?;
        let mut terms = Vec::new();
        for (dim, expr) in dims.iter().enumerate() {
            let stride = ((dim + 1)..dims.len())
                .map(|layout_dim| format!("{}.layout[{layout_dim}]", self.buffer_expr(buffer)))
                .collect::<Vec<_>>();
            if stride.is_empty() {
                terms.push(expr.clone());
            } else {
                terms.push(format!("({}) * {}", expr, stride.join(" * ")));
            }
        }
        Ok(terms.join(" + "))
    }

    fn lower_iter(&self, iter: &Iter) -> Result<String, RenderError> {
        match iter {
            Iter::Raw(loop_id) => self.loop_expr(*loop_id),
            Iter::Reconstructed { loops, factors } => {
                let iters = loops
                    .iter()
                    .copied()
                    .map(|loop_id| self.loop_expr(loop_id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(reconstruct_index(iters, factors))
            }
        }
    }

    fn loop_expr(&self, loop_id: LoopId) -> Result<String, RenderError> {
        self.loops
            .get(&loop_id)
            .map(|info| info.expr.clone())
            .ok_or_else(|| RenderError::new(format!("loop {} is not in scope", loop_id.0)))
    }

    fn reconstruct_extent_index(&self, source: DimRef<KernelRef>) -> Result<String, RenderError> {
        let semantic_source = self.semantic_dim(source);
        let mut infos = Vec::new();
        for loop_id in self.loop_stack.iter().rev().copied() {
            let info = self
                .loops
                .get(&loop_id)
                .ok_or_else(|| RenderError::new(format!("loop {} is not in scope", loop_id.0)))?;
            let matches = if let Some(source) = semantic_source {
                self.semantic_dim(info.source) == Some(source)
            } else {
                info.source == source
            };
            if !matches {
                continue;
            }
            infos.push(info.clone());
            if matches!(info.kind, ExtentKind::Base(_)) {
                break;
            }
        }
        infos.reverse();
        infos.sort_by_key(|info| extent_order(&info.kind));
        Ok(reconstruct_index(
            infos.iter().map(|info| info.expr.clone()).collect(),
            normalize_reconstruction_factors(
                infos
                    .iter()
                    .find_map(|info| match &info.kind {
                        ExtentKind::Base(factors) => Some(factors.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        infos
                            .iter()
                            .filter_map(|info| match info.kind {
                                ExtentKind::Split { factor, .. } => Some(factor),
                                _ => None,
                            })
                            .collect()
                    }),
                infos.len(),
            )
            .as_slice(),
        ))
    }

    fn semantic_dim(&self, source: DimRef<KernelRef>) -> Option<DimRef<Input>> {
        self.semantic_dims
            .iter()
            .find_map(|(param, input)| (*param == source).then_some(*input))
    }

    fn input_dim(&self, dim: DimRef<Input>) -> Result<String, RenderError> {
        self.semantic_dims
            .iter()
            .find_map(|(buffer, input)| (*input == dim).then_some(self.buffer_dim(*buffer)))
            .ok_or_else(|| {
                RenderError::new(format!(
                    "input {} dim {} is not available in kernel",
                    dim.buffer.0, dim.dim
                ))
            })
    }

    fn buffer_dim(&self, dim: DimRef<KernelRef>) -> String {
        format!("{}.shape[{}]", self.buffer_expr(dim.buffer), dim.dim)
    }

    fn buffer_expr(&self, buffer: KernelRef) -> String {
        match buffer {
            KernelRef::Param(param) => match param.arg {
                Arg::Readonly => format!("readonlys[{}]", param.ind),
                Arg::Writeable => format!("writeables[{}]", param.ind),
            },
            KernelRef::Local(local) => local_ident(local),
        }
    }

    fn zero_check_condition(&self, loops: &[LoopId]) -> Result<Option<String>, RenderError> {
        let conditions = loops
            .iter()
            .map(|loop_id| Ok(format!("{} == 0", self.loop_expr(*loop_id)?)))
            .collect::<Result<Vec<_>, RenderError>>()?;
        Ok((!conditions.is_empty()).then(|| conditions.join(" && ")))
    }

    fn push_loop_with_expr(
        &mut self,
        id: LoopId,
        expr: String,
        source: DimRef<KernelRef>,
        kind: ExtentKind,
    ) {
        self.loops.insert(id, LoopInfo { expr, source, kind });
        self.loop_stack.push(id);
    }

    fn pop_loop(&mut self, id: LoopId) {
        let popped = self.loop_stack.pop();
        debug_assert_eq!(popped, Some(id));
        self.loops.remove(&id);
    }
}

#[derive(Clone)]
struct LoopInfo {
    expr: String,
    source: DimRef<KernelRef>,
    kind: ExtentKind,
}

fn initialized_buffers(block: &Block<KernelRef>) -> BTreeSet<KernelRef> {
    let mut initialized = BTreeSet::new();
    collect_initialized(block, &mut initialized);
    initialized
}

fn collect_initialized(block: &Block<KernelRef>, initialized: &mut BTreeSet<KernelRef>) {
    for action in &block.0 {
        match action {
            Action::Loop { body, .. } => collect_initialized(body, initialized),
            Action::Init { write, .. } => {
                initialized.insert(write.buffer);
            }
            Action::Compute { .. } | Action::Snapshot { .. } | Action::Scale { .. } => {}
        }
    }
}

fn extent_expr(source: String, kind: &ExtentKind) -> String {
    match kind {
        ExtentKind::Semantic => source,
        ExtentKind::Base(factors) => {
            let factor = factors.iter().product::<usize>();
            if factor == 1 {
                source
            } else {
                format!("(({source} + {factor} - 1) / {factor})")
            }
        }
        ExtentKind::Split { factor, .. } => factor.to_string(),
    }
}

fn reconstruct_index(iters: Vec<String>, factors: &[usize]) -> String {
    if iters.is_empty() {
        return "0".to_string();
    }
    if factors.is_empty() {
        return iters.join(" + ");
    }
    let mut terms = Vec::new();
    for (index, iter) in iters.into_iter().enumerate() {
        let weight = if index < factors.len() {
            factors[index..].iter().product::<usize>()
        } else {
            1
        };
        if weight == 1 {
            terms.push(iter);
        } else {
            terms.push(format!("{iter} * {weight}"));
        }
    }
    terms.join(" + ")
}

fn normalize_reconstruction_factors(factors: Vec<usize>, loop_count: usize) -> Vec<usize> {
    let expected = loop_count.saturating_sub(1);
    if factors.len() > expected {
        factors[factors.len() - expected..].to_vec()
    } else {
        factors
    }
}

fn extent_order(kind: &ExtentKind) -> (usize, usize) {
    match kind {
        ExtentKind::Semantic => (0, 0),
        ExtentKind::Base(_) => (0, 0),
        ExtentKind::Split { level, .. } => (1, *level),
    }
}

fn init_value(op: Op) -> Result<String, RenderError> {
    Ok(match op {
        Op::Add | Op::Or | Op::Xor => "0.0f",
        Op::Mul | Op::And => "1.0f",
        Op::Max => "(-INFINITY)",
        Op::Min => "INFINITY",
        _ => {
            return Err(RenderError::new(format!(
                "op {:?} has no reduction identity",
                op
            )))
        }
    }
    .to_string())
}

fn render_op(op: Op, args: &[String]) -> Result<String, RenderError> {
    if args.len() == 1 {
        return Ok(match op {
            Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor => args[0].clone(),
            Op::Sub => format!("0.0f - {}", args[0]),
            Op::Div => format!("1.0f / {}", args[0]),
            Op::Max => format!("fmaxf(0.0f, {})", args[0]),
            Op::Min => format!("fminf(0.0f, {})", args[0]),
            Op::Pow => format!("powf(2.71828175f, {})", args[0]),
            Op::Log => format!("logf({})", args[0]),
            Op::Gt => format!("0.0f > {}", args[0]),
            Op::Ge => format!("0.0f >= {}", args[0]),
            Op::Lt => format!("0.0f < {}", args[0]),
            Op::Le => format!("0.0f <= {}", args[0]),
            Op::Eq => format!("0.0f == {}", args[0]),
            Op::Ne => format!("0.0f != {}", args[0]),
            Op::Not => format!("!{}", args[0]),
        });
    }
    Ok(match op {
        Op::Add => args.join(" + "),
        Op::Mul => args.join(" * "),
        Op::Div => args.join(" / "),
        Op::Sub => args.join(" - "),
        Op::Max => render_call_fold("fmaxf", args),
        Op::Min => render_call_fold("fminf", args),
        Op::Pow => render_call_fold("powf", args),
        Op::Log => {
            if args.len() != 1 {
                return Err(RenderError::new("log expects one argument"));
            }
            format!("logf({})", args[0])
        }
        Op::Gt => args.join(" > "),
        Op::Ge => args.join(" >= "),
        Op::Lt => args.join(" < "),
        Op::Le => args.join(" <= "),
        Op::Eq => args.join(" == "),
        Op::Ne => args.join(" != "),
        Op::And => args.join(" && "),
        Op::Or => args.join(" || "),
        Op::Xor => args.join(" != "),
        Op::Not => {
            if args.len() != 1 {
                return Err(RenderError::new("not expects one argument"));
            }
            format!("!{}", args[0])
        }
    })
}

fn render_call_fold(function: &str, args: &[String]) -> String {
    let mut iter = args.iter();
    let Some(first) = iter.next() else {
        return "0.0f".to_string();
    };
    iter.fold(first.clone(), |acc, arg| {
        format!("{function}({acc}, {arg})")
    })
}

fn static_layout_size(layout: &[String]) -> Option<usize> {
    if layout.is_empty() {
        return Some(1);
    }
    layout
        .iter()
        .map(|expr| expr.parse::<usize>().ok())
        .try_fold(1usize, |acc, value| value.map(|value| acc * value))
}

fn render_array_decl(ident: &str, values: Vec<String>) -> String {
    if values.is_empty() {
        format!("const size_t* {ident} = NULL;")
    } else {
        format!("const size_t {ident}[] = {{ {} }};", values.join(", "))
    }
}

fn array_arg(ident: &str, len: usize) -> String {
    if len == 0 {
        "NULL".to_string()
    } else {
        ident.to_string()
    }
}

fn indent_lines(lines: Vec<String>) -> String {
    lines
        .into_iter()
        .flat_map(|line| {
            line.lines()
                .map(|line| {
                    if line.is_empty() {
                        String::new()
                    } else {
                        format!("  {line}")
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn input_ident(input: Input) -> String {
    format!("in{}", input.0)
}

fn intermediate_ident(intermediate: Intermediate) -> String {
    format!("s{}", intermediate.0)
}

fn output_ident(output: crate::ir::exec_plan::Output) -> String {
    format!("out{}", output.0)
}

fn local_ident(local: Local) -> String {
    format!("l{}", local.0)
}

fn loop_ident(loop_id: LoopId) -> String {
    format!("i{}", loop_id.0)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderError {
    pub message: String,
}

impl RenderError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn from_parallel_module(error: crate::check::parallel_module::ValidationError) -> Self {
        Self::new(error.to_string())
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RenderError {}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::component;
    use crate::front::parse_expr;
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::exec_plan_to_parallel_module::lower_exec_plan_to_parallel_module;
    use crate::lower::kernel_program_to_exec_plan::lower_kernel_program_to_exec_plan;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;

    fn render_expr(src: &str) -> String {
        let component = component::expr(parse_expr(src).unwrap());
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        let module = lower_exec_plan_to_parallel_module(&exec_plan).unwrap();
        render(&module).unwrap()
    }

    #[test]
    fn renders_public_cuda_abi() {
        let cu = render_expr("i+i~i | i:256 | ii'");

        assert!(cu.contains("extern \"C\" size_t count(void)"));
        assert!(cu.contains("__global__ void f0"));
        assert!(cu.contains("dim3 f0_grid"));
        assert!(cu.contains("dim3 f0_block(256, 1, 1);"));
        assert!(cu.contains("f0<<<f0_grid, f0_block>>>"));
    }

    #[test]
    fn maps_three_execution_dimensions_to_grid_and_block() {
        let cu = render_expr("ijk+ijk~ijk|i:16,j:16,k:4|ii'jj'kk'");

        assert!(cu.contains("dim3 f0_grid("));
        assert!(cu.contains("dim3 f0_block(16, 16, 4);"));
        assert!(cu.contains("const size_t i0 = blockIdx.x;"));
        assert!(cu.contains("const size_t i2 = blockIdx.y;"));
        assert!(cu.contains("const size_t i4 = blockIdx.z;"));
        assert!(cu.contains("const size_t i1 = threadIdx.x;"));
        assert!(cu.contains("const size_t i3 = threadIdx.y;"));
        assert!(cu.contains("const size_t i5 = threadIdx.z;"));
    }

    #[test]
    fn parenthesizes_reconstructed_indices_before_stride_multiplication() {
        let cu = render_expr("ij+ij~ij | i:16,j:8 | ii'jj'");

        assert!(cu.contains("(blockIdx.x * 16 + threadIdx.x) * writeables[0].layout[1]"));
        assert!(!cu.contains("blockIdx.x * 16 + threadIdx.x * writeables[0].layout[1]"));
    }
}

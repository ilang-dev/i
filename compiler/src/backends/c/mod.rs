mod expr;
mod fmt;
mod stmt;

use crate::ir::module::{Ident, Module};

use stmt::render_function;

pub fn render(module: &Module) -> String {
    let kernels = module
        .kernels
        .iter()
        .map(render_function)
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        r#"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {{ const float* data; const size_t* shape; const size_t rank; }} Tensor;
typedef struct {{ float* data; const size_t* shape; const size_t rank; }} TensorMut;
typedef struct {{ const float* data; const size_t* shape; const size_t* layout; }} View;
typedef struct {{ float* data; const size_t* shape; const size_t* layout; }} ViewMut;

static inline ViewMut alloc_view_mut(size_t ndims, const size_t* layout, const size_t* shape) {{
  size_t n = 1;
  for (size_t i = 0; i < ndims; ++i) n *= layout[i];
  float* data = (float*)malloc(n * sizeof(float));
  return (ViewMut){{ .data = data, .shape = shape, .layout = layout }};
}}

static inline View as_view(const Tensor* t) {{
  return (View){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline ViewMut as_view_mut(const TensorMut* t) {{
  return (ViewMut){{ .data = t->data, .shape = t->shape, .layout = t->shape }};
}}

static inline View view_mut_as_view(const ViewMut* t) {{
  return (View){{ .data = t->data, .shape = t->shape, .layout = t->layout }};
}}

{count}

{ranks}

{shapes}

{kernels}

{exec}
"#,
        count = render_function(&module.count),
        ranks = render_function(&module.ranks),
        shapes = render_function(&module.shapes),
        kernels = kernels,
        exec = render_function(&module.exec),
    )
}

fn render_ident(ident: &Ident) -> String {
    ident.0.clone()
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::front::parse_expr;
    use crate::ir::module::{Block, Cast, Expr, Fn, Ident, Module, Signature, Stmt, Type};
    use crate::lower::component_to_graph::lower_component_to_graph;
    use crate::lower::exec_plan_to_module::lower_exec_plan_to_module;
    use crate::lower::kernel_program_to_exec_plan::lower_kernel_program_to_exec_plan;
    use crate::lower::node_to_stage::lower_node_graph_to_stage_program;
    use crate::lower::stage_to_kernel_program::lower_stage_program_to_kernel_program;
    use crate::{component, front};

    fn id(value: &str) -> Ident {
        Ident(value.to_string())
    }

    fn function(name: &str, signature: Signature, body: Block) -> Fn {
        Fn {
            ident: id(name),
            signature,
            body,
        }
    }

    fn render_expr(src: &str) -> String {
        let component = component::expr(parse_expr(src).unwrap());
        render_component(&component)
    }

    fn render_component(component: &crate::ir::component::Component) -> String {
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        let module = lower_exec_plan_to_module(&exec_plan).unwrap();
        render(&module)
    }

    #[test]
    fn renders_public_abi() {
        let c = render_expr("ik*kj~ijk");

        assert!(c.contains("size_t count(void)"));
        assert!(c.contains("void ranks(size_t* ranks)"));
        assert!(c.contains("void shapes(const Tensor* inputs, size_t** shapes)"));
        assert!(c.contains("void exec(const Tensor* inputs, TensorMut* outputs)"));
        assert!(c.contains("void f0(const View* readonlys, ViewMut* writeables)"));
    }

    #[test]
    fn renders_shapes_and_ranks() {
        let c = render_expr("+ij~ji");

        assert!(c.contains("ranks[0] = 2;"));
        assert!(c.contains("shapes[0][0] = inputs[0].shape[1];"));
        assert!(c.contains("shapes[0][1] = inputs[0].shape[0];"));
    }

    #[test]
    fn renders_exec_bindings_and_dispatch() {
        let c = render_expr("ik*kj~ijk");

        assert!(c.contains("const View in0 = as_view(&inputs[0]);"));
        assert!(c.contains("ViewMut out0 = as_view_mut(&outputs[0]);"));
        assert!(c.contains(
            "f0(\n    (const View[]){\n      in0,\n      in1\n    },\n    (ViewMut[]){\n      out0\n    }\n  );"
        ));
    }

    #[test]
    fn renders_intermediate_alloc_and_free() {
        let component = component::expr(front::parse_expr("ik*kj~ijk").unwrap())
            .chain(component::expr(front::parse_expr("+ijk~ij").unwrap()));
        let graph = lower_component_to_graph(&component).unwrap();
        let stage_program = lower_node_graph_to_stage_program(&graph).unwrap();
        let kernel_program = lower_stage_program_to_kernel_program(&stage_program).unwrap();
        let exec_plan = lower_kernel_program_to_exec_plan(&kernel_program).unwrap();
        let module = lower_exec_plan_to_module(&exec_plan).unwrap();
        let c = render(&module);

        assert!(c.contains("ViewMut s0 = alloc_view_mut("));
        assert!(c.contains(
            "f1(\n    (const View[]){\n      view_mut_as_view(&s0)\n    },\n    (ViewMut[]){\n      out0\n    }\n  );"
        ));
        assert!(c.contains("free(s0.data);"));
    }

    #[test]
    fn renders_split_bounds_and_positive_guards() {
        let c = render_expr("ik*kj~ijk|i:8|ii'jk");

        assert!(c.contains("for (size_t i0 = 0; i0 <"));
        assert!(c.contains("readonlys[0].shape[0] + 8"));
        assert!(c.contains("- 1) / 8"));
        assert!(c.contains("if ("));
        assert!(c.contains("i0 * 8"));
        assert!(c.contains("i1"));
        assert!(c.contains("< readonlys[0].shape[0]"));
        assert!(!c.contains("((((writeables"));
    }

    #[test]
    fn renders_fused_fanout_tail_guards_with_outer_tile_index() {
        let identity = crate::ir::component::Component::Identity;
        let normalize = identity
            .fanout(component::expr(
                front::parse_expr("+i~. | i:8 | ii'").unwrap(),
            ))
            .chain(component::expr(
                front::parse_expr("i/.~i | i:8 | i1i'").unwrap(),
            ));
        let dot = component::expr(front::parse_expr("i*i~i | i:8 | i0i'").unwrap()).chain(
            component::expr(front::parse_expr("+i~. | i:8 | ii'0").unwrap()),
        );
        let c = render_component(&normalize.chain(dot));

        assert!(c.contains("if (i0 * 8 + i3 < readonlys[0].shape[0])"));
        assert!(!c.contains("if (i3 * 8 < readonlys[0].shape[0])"));
    }

    #[test]
    fn renders_fused_root_reduction_init_before_consumer_tile_loop() {
        let identity = crate::ir::component::Component::Identity;
        let normalize = identity
            .fanout(component::expr(
                front::parse_expr("+i~. | i:8 | !ii'").unwrap(),
            ))
            .chain(component::expr(
                front::parse_expr("i/.~i | i:8 | i1i'").unwrap(),
            ));
        let dot = component::expr(front::parse_expr("i*i~i | i:8 | i0i'").unwrap()).chain(
            component::expr(front::parse_expr("+i~. | i:8 | !ii'0").unwrap()),
        );
        let c = render_component(&normalize.chain(dot));

        assert!(c.contains("l1.data[0] = 0.0f;\n  for (size_t i0 = 0;"));
        assert!(!c.contains("if (i0 == 0)"));
        assert!(c.contains(
            "writeables[0].data[0] = writeables[0].data[0] * (l0.data[0] / l1.data[0]);"
        ));
        assert!(c.contains("writeables[0].data[0] = 0.0f;"));
    }

    #[test]
    fn renders_rowwise_online_normalize_matmul_with_tile_indexing() {
        let identity = crate::ir::component::Component::Identity;
        let row_normalize = identity
            .fanout(component::expr(
                front::parse_expr("+ik~i | k:8 | kik'").unwrap(),
            ))
            .chain(component::expr(
                front::parse_expr("ik/i~ik | k:8 | k1ik'").unwrap(),
            ));
        let mm = component::expr(front::parse_expr("ik*kj~ijk | k:8 | k0ijk'").unwrap()).chain(
            component::expr(front::parse_expr("+ijk~ij | k:8 | kijk'0").unwrap()),
        );
        let c = render_component(&row_normalize.chain(mm));

        assert!(c.contains("for (size_t i0 = 0; i0 < (l0.shape[2] + 8 - 1) / 8; ++i0)"));
        assert!(c.contains("writeables[0].data[i6] = writeables[1].data[i6];"));
        assert!(c.contains("i0 * 8 +"));
        assert!(!c.contains("i5 * 8 +\n              i5"));
        assert!(c.contains("] * (writeables[0].data[i1] / writeables[1].data[i1]);"));
    }

    #[test]
    fn renders_rowwise_online_normalize_matmul_with_outer_row_tile_lifetime() {
        let identity = crate::ir::component::Component::Identity;
        let row_normalize = identity
            .fanout(component::expr(
                front::parse_expr("+ik~i | i:8,k:8 | kii'k'").unwrap(),
            ))
            .chain(component::expr(
                front::parse_expr("ik/i~ik | i:8,k:8 | ki1i'k'").unwrap(),
            ));
        let mm = component::expr(front::parse_expr("ik*kj~ijk | i:8,k:8 | ki0i'jk'").unwrap())
            .chain(component::expr(
                front::parse_expr("+ijk~ij | i:8,k:8 | kii'jk'0").unwrap(),
            ));
        let c = render_component(&row_normalize.chain(mm));

        assert!(c.contains(
            "] / writeables[1].data[\n                i1 * 8 +\n                i5\n              ];"
        ));
        assert!(!c.contains("] / writeables[1].data[i5];"));
        assert!(c.contains(
            "] * (writeables[0].data[\n              i1 * 8 +\n              i2\n            ] / writeables[1].data[\n              i1 * 8 +\n              i2\n            ]);"
        ));
        assert!(c.contains(
            "l0.data[\n                  i2 * l0.layout[1] +\n                  i4\n                ]"
        ));
    }

    #[test]
    fn renders_data_access_and_ops() {
        let c = render_expr("+ijk~ij");

        assert!(c.contains(".data[\n"));
        assert!(c.contains(".layout["));
        assert!(c.contains("= 0.0f;"));
        assert!(c.contains(" + "));
    }

    #[test]
    fn renders_normalized_unary_pow() {
        let c = render_expr("^ij~ij");

        assert!(c.contains("powf(2.718281"));
    }

    #[test]
    fn renders_unary_max_and_min_with_zero_defaults() {
        let max = render_expr(">ij~ij");
        let min = render_expr("<ij~ij");

        assert!(max.contains("fmaxf(0.0f, readonlys[0].data["));
        assert!(min.contains("fminf(0.0f, readonlys[0].data["));
    }

    #[test]
    fn renders_normalized_unary_comparisons_with_zero_defaults() {
        let gt = render_expr(">>ij~ij");
        let ge = render_expr(">=ij~ij");
        let lt = render_expr("<<ij~ij");
        let le = render_expr("<=ij~ij");
        let eq = render_expr("==ij~ij");
        let ne = render_expr("!=ij~ij");

        assert!(gt.contains("] > 0.0f;"));
        assert!(ge.contains("] >= 0.0f;"));
        assert!(lt.contains("] < 0.0f;"));
        assert!(le.contains("] <= 0.0f;"));
        assert!(eq.contains("] == 0.0f;"));
        assert!(ne.contains("] != 0.0f;"));
    }

    #[test]
    fn renders_zero_check_conjunction() {
        let c = render_expr("+ijkl~ij||ijkl!");

        assert!(c.contains(" == 0"));
        assert!(c.contains(" && "));
    }

    #[test]
    fn renders_dispatch_reads_from_bound_types() {
        let module = Module {
            count: function(
                "count",
                Signature::Count,
                Block(vec![Stmt::Return(Some(Expr::Usize(1)))]),
            ),
            ranks: function("ranks", Signature::Ranks, Block(vec![Stmt::Return(None)])),
            shapes: function("shapes", Signature::Shapes, Block(vec![Stmt::Return(None)])),
            exec: function(
                "exec",
                Signature::Exec,
                Block(vec![
                    Stmt::Let {
                        ident: id("source"),
                        ty: Type::ViewMut,
                        value: Expr::Cast {
                            kind: Cast::ViewMut,
                            value: Box::new(Expr::Index {
                                base: Box::new(Expr::Ident(id("outputs"))),
                                index: Box::new(Expr::Usize(0)),
                            }),
                        },
                    },
                    Stmt::Dispatch {
                        kernel: id("f0"),
                        reads: vec![id("source")],
                        writes: vec![id("source")],
                    },
                    Stmt::Return(None),
                ]),
            ),
            kernels: vec![function("f0", Signature::Kernel, Block(vec![]))],
        };
        let c = render(&module);

        assert!(c.contains(
            "f0(\n    (const View[]){\n      view_mut_as_view(&source)\n    },\n    (ViewMut[]){\n      source\n    }\n  );"
        ));
    }

    #[test]
    fn renders_stack_alloc() {
        let module = Module {
            count: function(
                "count",
                Signature::Count,
                Block(vec![Stmt::Return(Some(Expr::Usize(0)))]),
            ),
            ranks: function("ranks", Signature::Ranks, Block(vec![Stmt::Return(None)])),
            shapes: function("shapes", Signature::Shapes, Block(vec![Stmt::Return(None)])),
            exec: function("exec", Signature::Exec, Block(vec![Stmt::Return(None)])),
            kernels: vec![function(
                "f0",
                Signature::Kernel,
                Block(vec![Stmt::StackAlloc {
                    dst: id("l0"),
                    shape: vec![],
                    layout: vec![Expr::Usize(16), Expr::Usize(16)],
                }]),
            )],
        };
        let c = render(&module);

        assert!(c.contains("float l0_data[256];"));
        assert!(c.contains("const size_t l0_layout[] = {\n    16,\n    16\n  };"));
        assert!(c.contains("ViewMut l0 = (ViewMut){ .data = l0_data"));
    }
}

pub mod common;
pub mod component;
pub mod expr;
pub mod graph;
pub mod iir;
pub mod kernel;
pub mod kernel_program;
pub mod loop_ir;
pub mod node;
pub mod stage;

/*

- component holds i-exprs connected by combinators
- graph holds i-exprs connected by dataflow
- component -> graph resolves dataflow from combinators
- need to parse i-exprs into some sort of structured representation (stages?)
  - this parsing is essentially independent of the dataflow structure stuff
- ? could lower stages into loops at this point, could wait until kernel graph
- schedule-annotated-graph -> kernel graph
  - kernel graph is a dataflow graph over subgraphs (fused kernels)
- per kernel:
  - lower subgraphs

string -> expr -> stage -> loop

string: ik*kj~ijk
expr: *, [ik, kj], ijk
stage: *, [[0,2],[2,1]], [0,1,2]
loop: [Loop i0 N0, Loop i1 N1, Loop i2 N2] * out0[i0,i1,i2] [in0[i0,i2], in1[i2,i1]]

Component -> Stage Graph -> Kernel Graph

Component<T> -> Graph<T>

Component<String> -> Component<Expr>
Component<Expr> -> Graph<Expr>

Expr {
    op: Op,
    inputs: Vec<String>,
    output: String,
    schedule:
}

Expr {
    semantic_expr: {
        op: Op,
        inputs: Vec<Vec<char>>,
        outpus: Vec<char>,
    }
    splits: HashMap<char, Vec<usize>>,
    permutation: Vec<PermutationAtom>
}

PermutationAtom {
    Loop: {
        axis: char,
        index: usize,
    },
    InputIndex(usize),
}

Stage {
    op: Op,
    inputs: Vec<Vec<Axis>>,
    output: Vec<Axis>,
}


*/

// - component
// - graph
//   - semantic
//   - schedule annotated
//
// ???
//
// - loops
// - kernels
// - full program?

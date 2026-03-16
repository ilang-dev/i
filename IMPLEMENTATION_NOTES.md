# Implementation Notes

This document is for orchestrating parallel implementation work across the
compiler pipeline.

The canonical references are:

- `README.md` for language/operator semantics
- `ABI.md` for the exported C ABI
- `IR_LEVELS.md` for the level boundaries
- `c2/src/ir/*.rs` for the current IR types

The public IR stack is:

- `component`
- `semantic_graph`
- `scheduled_graph`
- `loop_ir`
- `iir`

There are also internal helper data structures that are worth having but should
not become public IR levels. The most important one is a kernel/execution plan
used to bridge `scheduled_graph` to `loop_ir` and `iir`.

## Global Rules

- Do not change the IR types casually. If a pass seems blocked on an IR issue,
  write down the minimal proposed IR change and why it is necessary.
- Treat `README.md` as the source of truth for operator semantics.
- Treat `ABI.md` as the source of truth for the exported C interface.
- Keep all passes deterministic. Stable ordering matters.
- Prefer rejecting ambiguous schedules to guessing.
- In the first implementation, do not do temp pooling or clever buffer reuse.
  One materialized temp value should get one temp buffer.
- Keep internal planning types private to lowering modules unless there is a
  very strong reason to expose them.

## Suggested Module Layout

This is only a suggestion, but it gives the agents a consistent target:

- `c2/src/front/parser.rs`
- `c2/src/check/component.rs`
- `c2/src/lower/component_to_semantic.rs`
- `c2/src/lower/semantic_to_shapes.rs`
- `c2/src/lower/component_semantic_to_scheduled.rs`
- `c2/src/check/scheduled_graph.rs`
- `c2/src/plan/kernel_plan.rs`
- `c2/src/lower/kernel_plan_to_loop.rs`
- `c2/src/lower/plan_to_iir.rs`
- `c2/src/backend/c.rs`
- `c2/tests/...`

## Dependency Graph

Recommended pass order:

1. `source -> component`
2. `component` validation
3. `component -> semantic_graph`
4. `semantic_graph -> iir::ShapeData`
5. `component + semantic_graph -> scheduled_graph`
6. `scheduled_graph` validation
7. `scheduled_graph -> internal kernel/exec plan`
8. `kernel plan -> loop_ir::Kernel`
9. `shape data + kernel plan + loop kernels -> iir::Program`
10. `iir -> C`
11. end-to-end pipeline + oracle tests

Parallelism:

- 4 and 5 can proceed after 3.
- 8 and 9 can be worked on in parallel once 7 is stable enough.
- 10 can start once 9 is stable.
- 11 should start early with scaffolding, then keep expanding.

## Internal Planning Data

It is worth introducing a private planning layer under `c2/src/plan/` with
types roughly like:

```rust
struct KernelPlan {
    kernel: usize,
    stages: Vec<usize>,
    reads: Vec<usize>,
    writes: Vec<usize>,
    placements: Vec<StagePlacement>,
}

struct StagePlacement {
    stage: usize,
    at: Vec<super::ir::common::AxisRef>,
}

struct ExecPlan {
    buffers: Vec<BufferPlan>,
    calls: Vec<CallPlan>,
}

struct BufferPlan {
    buffer: usize,
    value: usize,
    ty: super::ir::common::TensorType,
    kind: BufferKind,
}

enum BufferKind {
    Output(usize),
    Temp,
}

struct CallPlan {
    kernel: usize,
    reads: Vec<usize>,
    writes: Vec<usize>,
}
```

Do not treat these exact names as fixed. The point is that kernel planning is a
real implementation concern, but it is not itself a public IR level.

## Pass 1: Parse Source to Component

Input -> Output:

- `&str -> component::Component`

Responsibilities:

- Tokenize source syntax.
- Parse `i` expressions and combinators.
- Parse schedule syntax.
- Assign `ExprId` in stable left-to-right source order of `i` expressions.
- Preserve source-shaped structure exactly.

### Test Prompt

```text
Read README.md, IR_LEVELS.md, and c2/src/ir/{common,component}.rs.

Build a comprehensive parser test suite for source -> component. Cover:
- every operator spelling listed in README.md
- pointwise binary, pointwise unary, and reduction forms
- combinators: compose, chain, fanout, pair, swap
- precedence and associativity rules
- parentheses and nesting
- schedule syntax: splits, order, compute-at
- multi-expression programs composed with combinators
- malformed syntax with clear expected failures

Add strong assertions on the exact component tree shape and on ExprId
assignment order. Keep tests table-driven where possible. Add a few larger
integration-shaped parse tests that resemble realistic programs such as row
normalization and partial matrix product.

Do not implement the parser in this task unless a tiny helper is required to
make the tests compile.
```

### Implementation Prompt

```text
Implement source -> component parsing in a small, deterministic frontend
module. Read README.md for the language surface and c2/src/ir/component.rs for
the target IR.

Requirements:
- return component::Component
- assign ExprId in stable left-to-right source order of source i-expressions
- preserve combinator structure exactly
- parse schedule syntax into component::Schedule without trying to resolve it
  semantically yet
- produce useful syntax errors

Keep the parser small and explicit. Avoid backtracking-heavy designs if a
simple recursive-descent parser is enough. Do not validate deep semantics here;
that belongs in later passes.
```

## Pass 2: Validate Component

Input -> Output:

- `&component::Component -> Result<()>`

Responsibilities:

- Check expression-local well-formedness.
- Check operator forms against README semantics.
- Check schedule references against the local expression.

### Test Prompt

```text
Read README.md and c2/src/ir/component.rs. Build a robust validation test suite
for component IR.

Cover:
- valid and invalid operator arities/forms
- valid and invalid reduction forms
- valid and invalid schedule references
- duplicate or contradictory split entries
- order references to nonexistent split parts
- compute_at vector length mismatches
- edge cases around repeated axis names in inputs and outputs

The tests should clearly separate parser acceptance from semantic validation.
Something can parse successfully and still fail component validation.
```

### Implementation Prompt

```text
Implement component validation as a pure check pass over component::Component.

The validator should:
- enforce operator-form legality from README.md
- check that schedule references are local and syntactically meaningful for the
  expression they belong to
- reject malformed split/order/compute_at combinations
- remain agnostic about graph-level fusion/materialization questions

Keep the validator deterministic and give precise error messages. Do not
resolve schedule onto stages here.
```

## Pass 3: Lower Component to Semantic Graph

Input -> Output:

- `&component::Component -> semantic_graph::Graph`

Responsibilities:

- Eliminate combinators.
- Produce one semantic stage per source `component::Expr`.
- Build graph inputs, graph outputs, stage uses, and value numbering.
- Convert source-pattern syntax into explicit stage-local indexing.
- Construct stage-local axes with deterministic ordering.

Deterministic policy:

- each source `ExprId` yields exactly one semantic stage
- graph input `ValueId`s come first
- stage result `ValueId`s follow in stage order
- stage axis order should be:
  - output axes first, in output order
  - then any remaining input-only axes in first-appearance order scanning
    inputs left-to-right

### Test Prompt

```text
Read IR_LEVELS.md, c2/src/ir/component.rs, and c2/src/ir/semantic_graph.rs.
Write a comprehensive test suite for lowering component -> semantic graph.

Cover:
- pure pointwise expressions
- broadcasting cases such as ij+i~ij
- reductions such as +ijk~ij
- repeated cross-input indices such as ik*kj~ijk
- multi-output graphs produced by pair/fanout
- compose/chain wiring behavior
- swap behavior
- stable stage.expr mapping
- stable ValueId numbering
- stage-local axis ordering policy
- output index derivation

Add assertions that no source Pattern or source-axis chars survive as semantic
indexing structure. The semantic graph should be explicit and stage-local.
```

### Implementation Prompt

```text
Implement lowering from component IR to semantic graph IR.

Requirements:
- eliminate combinators into a pure dataflow graph
- emit exactly one semantic stage per source component::Expr
- preserve the source ExprId on semantic_graph::Stage.expr
- assign graph input and stage result ValueIds deterministically
- convert source patterns into explicit semantic_graph::Index values over
  stage-local axes
- compute stage-local axes in a fixed canonical order:
  output axes first, then remaining input-only axes in first-appearance order

Use small helper functions for:
- wiring combinators
- collecting stage-local axes
- converting a source pattern to a semantic index

Keep the resulting graph pure: no schedule information belongs here.
```

## Pass 4: Lower Semantic Graph to Shape Data

Input -> Output:

- `&semantic_graph::Graph -> iir::ShapeData`

Responsibilities:

- Compute output count, ranks, element scalar types, and output dimension
  expressions.
- This is the shape-side lowering for `count`, `ranks`, and `shapes`.
- This pass must be independent of schedule.

### Test Prompt

```text
Read ABI.md, IR_LEVELS.md, c2/src/ir/semantic_graph.rs, and c2/src/ir/iir.rs.
Write tests for semantic_graph -> iir::ShapeData.

Cover:
- single-output programs
- multi-output programs
- output rank derivation
- constant extents
- extents that come directly from input dimensions
- broadcasting and reduction effects on output shape
- logical/comparison ops producing Int outputs
- schedule-independence: shape data must not depend on any scheduling choice

If the current Extent representation needs an internal helper table to recover
InputDim expressions cleanly, the tests should make that requirement explicit.
```

### Implementation Prompt

```text
Implement semantic_graph -> iir::ShapeData.

Requirements:
- derive output scalar types from op semantics
- derive output rank from semantic output indices
- derive each output dimension as either:
  - DimExpr::Const
  - DimExpr::InputDim
- remain completely independent of scheduled_graph and loop_ir

If semantic extents need a small internal normalization table to map symbolic
extents back to concrete input dimensions, implement that as a private helper.
Do not add a new public IR level for shape inference.
```

## Pass 5: Resolve Schedule into Scheduled Graph

Input -> Output:

- `(&component::Component, &semantic_graph::Graph) -> scheduled_graph::Graph`

Responsibilities:

- Translate source schedule syntax into canonical stage-attached schedule data.
- Build default schedules where the source left things empty.
- Resolve fusion intent from `compute_at`.

Deterministic default policy:

- `splits = []`
- `order =` unsplit semantic stage axes in stage axis order
- `compute_at = Root`
- `store_at = Some(Root)`
- `fusions =` singleton stage groups

Important rule:

- if a producer is requested to compute at conflicting sites through different
  consumers, reject the schedule rather than guessing

### Test Prompt

```text
Read c2/src/ir/{component,semantic_graph,scheduled_graph}.rs and IR_LEVELS.md.
Write a detailed test suite for schedule resolution.

Cover:
- empty/default schedules
- simple split/order resolution
- compute_at resolution from consumer-local syntax to producer stage placement
- singleton fusion vs fused producer-consumer chains
- invalid compute_at targeting a graph input
- invalid references to nonexistent stage axes/split parts
- conflicting compute_at requests on a shared producer
- deterministic fusion-group construction
- deterministic default store_at policy

The tests should assert both the per-stage schedules and the fusion groups.
```

### Implementation Prompt

```text
Implement schedule resolution from (component, semantic_graph) to
scheduled_graph.

Requirements:
- map each source expression schedule onto the semantic stage with matching
  Stage.expr
- build canonical StageSchedule values
- synthesize default schedules when the source schedule is empty
- derive fusion groups from compute_at relationships
- assign a single compute_at/store_at policy per stage
- reject conflicts explicitly

Keep this pass pure and deterministic. The output is still a declarative
scheduled graph, not an executable loop tree.
```

## Pass 6: Validate Scheduled Graph

Input -> Output:

- `&scheduled_graph::Graph -> Result<()>`

Responsibilities:

- Check schedule legality after canonicalization.
- Catch impossible or contradictory schedule states before loop lowering.

### Test Prompt

```text
Read c2/src/ir/scheduled_graph.rs and write a strong validation suite.

Cover:
- invalid split references
- duplicate stages across fusion groups
- missing stages from fusion groups if your implementation requires full
  coverage
- invalid compute_at/store_at placements
- cycles or contradictions induced by compute/store relationships
- fusion groups that violate producer-consumer dominance assumptions

Prefer tests that assert exact error cases, not just generic failure.
```

### Implementation Prompt

```text
Implement scheduled-graph validation.

The validator should check:
- stage schedules are internally coherent
- placements reference real stages and real axis refs
- fusion groups are well-formed
- no contradictions remain that would make loop lowering ambiguous

This pass should be strict. Loop lowering should not have to guess what a bad
scheduled graph means.
```

## Pass 7: Build Internal Kernel and Exec Plans

Input -> Output:

- `&scheduled_graph::Graph -> (KernelPlanSet, ExecPlan)`

Responsibilities:

- Partition scheduled stages into kernels using fusion groups.
- Decide kernel call order.
- Decide which semantic values materialize into writable buffers.
- Decide read/write slot ordering per kernel.
- Decide temp buffer lifetimes for the eventual `exec`.

First-version policy:

- one materialized non-output value gets one temp buffer
- no temp pooling
- free temps immediately after last use
- stable ordering everywhere

### Test Prompt

```text
Read scheduled_graph IR and the current loop_ir/iir IRs. Write tests for the
internal kernel/exec planning pass.

Cover:
- one-stage one-kernel programs
- simple chains with one temp
- fused chains that produce one kernel
- fanout/shared producers requiring one materialized temp
- pair/multi-output programs
- topological kernel call ordering
- deterministic read slot ordering
- deterministic write slot ordering
- temp free-after-last-use behavior

For now, assume no temp pooling. The tests should lock in that simple policy.
```

### Implementation Prompt

```text
Implement a private planning pass from scheduled_graph to internal kernel and
exec plans.

Requirements:
- use fusion groups to form kernel boundaries
- topologically order kernels
- assign read slots and write slots deterministically for each kernel
- identify which values are final outputs vs temps
- build an ExecPlan with one temp buffer per materialized non-output value
- free each temp after its last use

Keep these planning types private. They are implementation glue, not new public
IR levels.
```

## Pass 8: Lower Kernel Plans to Loop IR

Input -> Output:

- `&KernelPlan -> loop_ir::Kernel`

Responsibilities:

- Turn one planned kernel into an explicit loop tree.
- Emit stage actions with explicit read/write accesses.
- Make reductions explicit as `Init` and `Update`.
- Reconstruct logical indices from scheduled loop structure.

### Test Prompt

```text
Read c2/src/ir/loop_ir.rs and the scheduled graph types. Write a strong test
suite for lowering a kernel plan to loop IR.

Cover:
- simple pointwise kernels
- simple reduction kernels with explicit Init and Update
- fused producer-consumer kernels
- kernels with repeated indices across inputs
- read/write slot usage
- loop ordering after split/reorder
- index reconstruction using LinearExpr
- action ordering that guarantees producer-before-consumer dominance

Tests should assert the exact loop/body shape, not just broad properties.
```

### Implementation Prompt

```text
Implement kernel-plan -> loop_ir lowering.

Requirements:
- emit a single loop_ir::Kernel per KernelPlan
- build an explicit nested loop tree
- place stage actions at the planned loop sites
- emit ActionKind::{Compute, Init, Update} correctly
- use loop_ir::Access over read/write slots only
- reconstruct indices with LinearExpr deterministically

Do not allocate buffers here. Loop IR is a kernel-body language only. If
something feels like whole-program ownership, it belongs in iIR planning, not
here.
```

## Pass 9: Assemble iIR

Input -> Output:

- `(&iir::ShapeData, &ExecPlan, &[loop_ir::Kernel]) -> iir::Program`

Responsibilities:

- Package loop-IR kernels into a whole program.
- Attach shape data for `count`, `ranks`, and `shapes`.
- Emit `exec` steps for alloc, call, and free.

### Test Prompt

```text
Read c2/src/ir/iir.rs and write tests for iIR assembly.

Cover:
- shape data plumbed through unchanged
- kernel list ordering
- exec alloc/call/free sequencing
- call-site read/write buffer bindings
- final outputs vs temp buffers
- multi-kernel programs
- multi-output programs

The tests should assert that iIR is now the whole program again, not just one
kernel body.
```

### Implementation Prompt

```text
Implement iIR assembly from shape data, exec plan, and loop IR kernels.

Requirements:
- build iir::Program
- preserve ShapeData exactly
- insert kernels in stable order
- generate exec steps from the ExecPlan
- bind loop_ir read/write slots to concrete BufferIds at each call site

Keep iIR simple. It should be almost entirely explicit orchestration and shape
metadata.
```

## Pass 10: Render iIR to C

Input -> Output:

- `&iir::Program -> String`

Responsibilities:

- Render the exported C library defined by `ABI.md`.
- Emit:
  - `count`
  - `ranks`
  - `shapes`
  - `exec`
  - internal kernel helpers

### Test Prompt

```text
Read ABI.md and c2/src/ir/iir.rs. Build a backend test suite for iIR -> C.

Cover:
- exact exported function signatures
- count/ranks/shapes behavior
- simple exec bodies
- multi-kernel programs
- multi-output programs
- temp allocation/free emission
- internal kernel helper signatures and slot plumbing
- representative loop rendering cases

Prefer string-structure tests plus a smaller number of compile-and-run tests if
the environment allows invoking a C compiler in CI.
```

### Implementation Prompt

```text
Implement the C backend from iIR.

Requirements:
- obey ABI.md exactly for exported functions
- render shape functions from iir::ShapeData
- render exec from explicit alloc/call/free steps
- render loop_ir kernels as internal helper functions
- use flat affine indexing in emitted C

Keep the backend mostly mechanical. The hard reasoning should already be done
above iIR.
```

## Pass 11: End-to-End Pipeline and Oracle Tests

Input -> Output:

- whole compiler pipeline tests

Responsibilities:

- Catch cross-pass regressions.
- Provide confidence that the IR boundaries compose correctly.

### Test Prompt

```text
Build an end-to-end test suite for the compiler pipeline.

Read README.md, ABI.md, IR_LEVELS.md, and the IR definitions first. Then:
- create a small reference interpreter at the semantic-graph level, or another
  trustworthy oracle, for value semantics
- add shape-oracle checks for count/ranks/shapes
- run representative programs through the full pipeline
- compare both values and shapes against the oracle

Cover:
- pointwise arithmetic
- broadcasting
- reductions
- logical/comparison ops
- chained expressions
- fanout/pair/multi-output graphs
- fused and unfused schedules
- at least one realistic example such as row normalization or partial matrix
  product
```

### Implementation Prompt

```text
Implement the end-to-end compiler driver and the test harness that exercises
the full pipeline.

Requirements:
- wire together all passes in order
- keep errors precise and phase-specific
- add integration tests that compile and execute representative programs
- compare against a small oracle implementation rather than trusting emitted C
  blindly

The goal is confidence in the full lowering story, not just unit-test coverage
of isolated helpers.
```

## Final Notes for Agents

- The most important invariant in the whole system is that semantic graph stays
  pure.
- The most important implementation seam is the private kernel/exec planning
  pass between scheduled graph and the lower executable layers.
- The most likely place for future pressure is `loop_ir::LinearExpr`. Do not
  widen it casually, but if a pass truly needs richer expressions, document the
  exact missing case with a concrete example before changing it.

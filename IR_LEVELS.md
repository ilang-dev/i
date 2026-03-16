# IR Levels

The compiler pipeline should be organized into five real levels. Each level
should answer one new class of questions and refuse to answer the next one
early.

The main design rule is:

- semantic meaning and schedule meaning must remain separate until the compiler
  is forced to join them

That gives the following stack:

- `source`
- `component`
- `semantic graph`
- `scheduled graph`
- `loop IR`
- `iIR`

## 1. Component IR

This is the parsed user program in its native form: `i` expressions plus
combinators. It is still source-shaped.

Contents:
- `Component`
- source `Expr`
- source patterns
- source schedule syntax

Strong invariants:
- The program is syntactically valid.
- Combinator arities are resolved.
- User-written schedule syntax is parsed, but not yet trusted.
- Source structure is preserved.
- Each source `i` expression has a stable identity.

This level is front-end structure, not semantics yet.

## 2. Semantic Graph IR

This is the first serious IR. Combinators are gone. What remains is a pure
dataflow graph of semantic stages.

Each stage corresponds to exactly one source `i` expression.

Each stage says:
- which source expression it came from
- which op it performs
- which values it reads
- what its local iteration axes are
- how each input is indexed by those axes
- how its output is indexed by those axes

Important point:
- this level is no longer written in source-pattern syntax
- indexing is explicit and stage-local
- axis chars do not survive as semantic indexing structure

Strong invariants:
- The graph is purely declarative: no split, no reorder, no fusion, no
  placement.
- Every stage has explicit local iteration axes.
- Every use-def edge is explicit.
- Every input access and output access is explicit.
- Reduction axes are derivable as stage axes that do not appear in the output
  index.
- Result type is derivable from the op.
- Equivalent source programs with different combinator spellings lower to the
  same semantic graph.

This is the protected mathematical core of the compiler.

## 3. Scheduled Graph IR

This is still a whole-program graph of semantic stages, but now the schedule is
attached in canonical form.

Contents:
- the semantic graph
- one schedule per stage
- fusion groups

Each stage schedule says:
- how its axes are split
- in what order those split axes appear
- where it computes
- where it stores, if materialized

Strong invariants:
- Semantic meaning is unchanged from semantic graph IR.
- Schedule legality has already been checked.
- Schedule refers to canonical stage identities.
- Fusion is represented as a grouping over stages, not yet as emitted loops.
- This is still declarative. It says what execution structure is intended, not
  yet the exact imperative body.

There is no separate public "resolved" IR. The step from component schedule
syntax to canonical schedule attachment is part of constructing scheduled graph
IR.

## 4. Loop IR

This is where the representation stops being graph-first and becomes a
structured executable body.

Loop IR does not represent a whole program. It represents one kernel body.

Contents:
- nested loops
- stage actions
- explicit read/write accesses
- explicit loop-index expressions

A loop-IR kernel sees only:
- read-only slots
- writable slots

It does not know whether a writable slot is a final output or a temporary.
That distinction belongs to whole-program orchestration, not to the kernel
body.

Strong invariants:
- Each loop-IR kernel corresponds to one scheduled kernel body.
- Every semantic stage instance has a unique placement in the loop tree.
- Every reduction has explicit `init` and `update` actions.
- Read and write order is explicit.
- No unresolved fusion or compute-at questions remain.
- Buffer accesses are explicit in loop space.

This is the level of imperative execution, not of program ownership.

## 5. iIR

This is the whole exported program, ready for backend rendering.

It contains:
- shape data for `count`, `ranks`, and `shapes`
- a kernel library, where each kernel body is written in loop IR
- one `exec` procedure that allocates temporaries, binds buffers to kernel
  slots, calls kernels, and frees temporaries

This is the first level that fully represents the public generated program
again.

Strong invariants:
- Kernel boundaries are fixed.
- Call sites are fixed.
- Temporary allocation and lifetime are explicit.
- `count`, `ranks`, and `shapes` are derivable from explicit shape data.
- `exec` sequencing is explicit.
- Backends should mostly render, not reason.

Important point:
- shape inference belongs to the semantic side of the compiler
- schedule does not affect `count`, `ranks`, or `shapes`
- value execution and shape execution meet again only in `iIR`

## Level Boundaries

The right mental model is:

- `component` represents the source-shaped program
- `semantic graph` represents the full pure meaning of the program
- `scheduled graph` represents the full scheduled meaning of the program
- `loop IR` represents one extracted executable kernel body
- `iIR` represents the full generated program again

So not every level has to be a single monolithic whole-program object of the
same kind. It is normal for a middle level to be a per-kernel body language,
as long as a later level wraps those bodies back into a whole program.

## Reduction Philosophy

Reductions should be treated as:
- one semantic stage in semantic graph IR
- one scheduled stage in scheduled graph IR
- two imperative actions (`init` and `update`) in loop IR

That keeps the math clean while making execution unambiguous.

## Minimality Test

A level is justified only if it freezes a new category of decisions:

- Component IR: source structure
- Semantic Graph IR: mathematical meaning
- Scheduled Graph IR: legal schedule attachment
- Loop IR: imperative kernel-body execution
- iIR: whole-program orchestration and exported shape/value interface

If a level cannot be defined by a new frozen decision class, it probably should
not exist.

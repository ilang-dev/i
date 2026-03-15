# IR Levels

The compiler pipeline should be organized into five real levels, with one hard
rule: each level should answer one new class of questions and forbid itself
from answering the next one early.

The main design principle is this: do not let the first real graph IR mix
algorithm semantics with schedule semantics. "What is computed" and "when/where
it is computed" should remain separate until the compiler is forced to join
them.

## 1. Component IR

This is the parsed user program in its native form: `i` expressions plus
combinators. Nodes are still expression instances and combinator applications.
This level preserves source structure.

Strong invariants:
- The program is syntactically valid.
- Combinator arities are resolved.
- User-written schedule syntax is parsed, but not yet trusted.
- No graph rewrites have happened yet.
- Source names and source order are preserved for diagnostics.

This level is mostly front-end machinery. It should not be an optimization
level.

## 2. Semantic Graph IR

This is the first serious IR. Combinators are gone. What remains is a pure DAG
of semantic stages. Each stage says: inputs, output value, iteration domain,
access/indexing maps, reduction axes if any, and result type/shape.

Strong invariants:
- The graph is purely declarative: no loop order, no fusion, no placement.
- Every stage has explicit domain semantics.
- Every use-def edge is explicit.
- Broadcasting and reduction behavior are explicit, not implicit in string
  syntax.
- Shape, rank, and type are consistent.
- Equivalent source programs with different combinator spellings lower to the
  same or nearly the same form.

This is where the math lives. If one IR deserves the strongest protection, it
is this one.

## 3. Scheduled Graph IR

This is still a graph of semantic stages, but now schedule decisions are
attached to stages and stage relationships. This is where split, reorder,
compute-at, fusion, and storage-folding intent live.

Strong invariants:
- Semantic meaning is unchanged from Semantic Graph IR.
- Every schedule directive refers to canonical stage ids and canonical axes.
- Schedule legality is already checked.
- Fusion is represented as a relation or grouping over stages, not yet as
  emitted loops.
- Storage intent is explicit, but concrete buffers do not yet need final
  identities.

This level should remain declarative. It answers "what execution structure is
intended?" but not yet "what exact imperative program gets emitted?"

## 4. Kernel Loop IR

This is where the representation stops being graph-first and becomes a
tree/region model. A kernel body is not naturally a graph. It is a nested loop
tree with stage actions inside regions.

This is the right home for the previously fuzzy "loop level". It should be a
loop tree, not a flat graph.

Contents:
- Explicit loop nests.
- Explicit stage placement.
- Explicit init/update actions for reductions.
- Explicit allocations and buffer views.
- Exact lifetimes and storage regions.
- Exact read/write order inside a kernel.

Strong invariants:
- Each kernel is a closed scheduled subgraph.
- Every stage instance has a unique placement in the loop tree.
- Every reduction has explicit init and update actions.
- Every temporary has a concrete allocation site and lifetime region.
- Every read is dominated by the writes it depends on.
- No unresolved fusion or compute-at questions remain.

This is the first level where allocation truly belongs, because allocation is a
lifetime question, and lifetime only becomes unambiguous once placement is
fixed.

## 5. iIR Program

This is the whole program IR, ready for backend rendering. It contains kernel
definitions plus the host or executive logic that allocates top-level buffers,
launches or calls kernels, and sequences them.

Strong invariants:
- The program is fully explicit and executable.
- Kernel boundaries are fixed.
- Call sites are fixed.
- Buffer ABI, argument order, and result plumbing are fixed.
- Backends should mostly render, not reason.

Kernel IR and iIR are close, but they should still remain conceptually
distinct. Kernel IR is one executable kernel body. iIR is the complete program
made of kernels plus orchestration.

## Direct Answers to Open Questions

The current "graph level" should not include schedule. It should be split into:
- Semantic Graph IR
- Scheduled Graph IR

The current "loop level" should not have the same structure as the graph. It
should be a structured loop tree with regions or statements. Execution is
nested and ordered; graphs are better for semantics and dependence, not final
execution structure.

Tensor indexing does not need its own whole level yet. It should remain part of
semantic stage definitions as access maps. A separate indexing IR only becomes
worthwhile later if the language grows real affine transforms, gather/scatter,
layout transforms, or symbolic simplification that deserves dedicated passes.

Allocations belong at Kernel Loop IR, not earlier.

## Reduction Philosophy

Reductions should be treated as:
- One semantic stage in Semantic Graph IR.
- One scheduled stage in Scheduled Graph IR.
- Two imperative actions (`init` and `update`) in Kernel Loop IR.

That keeps the mathematics clean while making execution unambiguous.

## Minimality Test

A level is justified only if it freezes a new category of decisions:
- Component IR: source structure
- Semantic Graph IR: mathematical meaning
- Scheduled Graph IR: legal schedule intent
- Kernel Loop IR: imperative execution and lifetime
- iIR Program: whole-program orchestration and backend-ready form

If a level cannot be defined by a new frozen decision class, it probably should
not exist.

This is the current recommended minimal solid stack.

# Component To Graph

This document specifies the transformation
`Component -> Graph<ScheduledStage>`.

## Leaves

- `Component::Expr(expr)` lowers to a graph with:
  - one graph input per expression input
  - one node containing `lower_expr_to_stage(expr)`
  - one node output
  - one graph output naming that node output

## Node Outputs

- Every scheduled-stage node has one output.

## Pair

- `Pair(left, right)` concatenates:
  - left inputs, then right inputs
  - left nodes, then right nodes
  - left outputs, then right outputs

## Chain

- `Chain(left, right)` pairs left outputs to right inputs left-to-right.
- The number of paired values is `min(left.outputs.len(), right.inputs.len())`.
- Each paired right input is replaced by the corresponding left output.
- Unpaired right inputs become graph inputs after all left inputs.
- Unpaired left outputs remain graph outputs before all right outputs.
- Node order is left nodes, then right nodes.

## Compose

- `Compose(left, right)` pairs right outputs to left inputs left-to-right.
- The number of paired values is `min(left.inputs.len(), right.outputs.len())`.
- Each paired left input is replaced by the corresponding right output.
- Unpaired left inputs become graph inputs after all right inputs.
- Unpaired right outputs remain graph outputs after all left outputs.
- Node order is right nodes, then left nodes.

## Fanout

- `Fanout(left, right)` pairs left inputs and right inputs left-to-right.
- The number of paired values is `min(left.inputs.len(), right.inputs.len())`.
- Each paired right input is replaced by the corresponding left graph input.
- Unpaired right inputs become graph inputs after all left inputs.
- Node order is left nodes, then right nodes.
- Graph outputs are left outputs, then right outputs.

## Swap

- `Swap(inner)` swaps the first two graph outputs if they exist.
- Otherwise it leaves the graph unchanged.

## Ordering

- Graph input order, node order, and graph output order are determined recursively by the rules above.

## Validation

- The input component is validated before lowering.
- The lowered graph is validated after lowering.

## Source Of Truth

- `src/lower/component_to_graph.rs`
- `src/check/component.rs`
- `src/check/graph.rs`

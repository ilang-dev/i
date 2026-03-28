# Kernelization

This document specifies the transformation
`Graph<ScheduledStage> -> Graph<Kernel>`.

## Terms

- A program output is an entry in `Graph<ScheduledStage>.outputs`.
- A materializing use is either a program output or a consumer input with `compute_sites[i] == None`.

## Compute-At

- `None` means the input value is produced outside the consumer kernel.
- `Some(site)` means the input value is produced inside the consumer kernel at `site`.

## Kernel Formation

- Each distinct stage result named by a program output or any other materializing use seeds one kernel.
- A program output `Source::Input(_)` does not seed a kernel.
- A kernel is the recursive closure of its seed through consumer inputs whose `compute_sites[i]` is `Some(_)`.
- Distinct program outputs do not fuse into the same kernel.

## Duplication

- A `Some(_)` use duplicates per consumer edge.
- Materializing uses do not cause duplication by themselves.
- All materializing uses of the same stage result share one materialized instance.

## Program Inputs

- A graph input may not be named by a consumer input whose `compute_sites[i]` is `Some(_)`.

## Interfaces

- Kernel inputs are the deduplicated boundary values of the kernel.
- Kernel outputs are all ordered stage-instance outputs of the kernel.
- Program output order is preserved.

## Ordering

- Kernel graph nodes are ordered by original stage-node order of their materialized roots.
- Kernel inputs are ordered by first encounter during DFS, in operand order.
- Within a kernel, stage instances are ordered in DFS postorder.
- Kernel outputs are in the same order.

## Validation

- A scheduled-stage graph node has one output.
- A scheduled-stage graph node has one graph input per stage input.
- In a kernel body, a node input sourced from `Input(_)` has `compute_sites[i] == None`.
- In a kernel body, a node input sourced from `Node(_, _)` has `compute_sites[i] == Some(_)`.
- A kernel-graph node input arity matches its inner kernel input arity.
- A kernel-graph node output arity matches its inner kernel output arity.

## Source Of Truth

- `src/lower/graph_to_kernel_graph.rs`
- `src/check/kernel.rs`
- `src/check/graph.rs`

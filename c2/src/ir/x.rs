/*

we have a graph of (nodes)
- plan out all the buffers
  - this can't be done without duplications
    - can plan all the "semantic" buffers though

- plan all semantic buffers
  - one per input plus one per interior node
- plan all semantic loops
- handle splits and ordering?
- handle fusion and somehow kernels?

*/

// ScheduledStage -> Plan

// when going from stage::AxisRef -> Extent, we lose the "part" of the axis
// which seems useful to track


// loop extents in terms of axes
// this is basically just order where AxisRef -> Extent<Axis>

/*
[
    extent: Extent<Axis> {
        source: Axis(0),
        kind: ExtentKind::Semantic,
    },
    extent: Extent<Axis> {
        source: Axis(1),
        kind: ExtentKind::Semantic,
    },
    extent: Extent<Axis> {
        source: Axis(2),
        kind: ExtentKind::Split([2]),
    },
    extent: Extent<Axis> {
        source: Axis(2),
        kind: ExtentKind::Factor(2),
    },
]
*/


struct Extent {
    source: ExtentSource,
    kind: ExtentKind,
}

enum ExtentSource {
    Input(DimRef<Input>), // graph-level input
    Operand(DimRef<Operand>), // expr-level operand
    Param(DimRef<Param>), // kernel param
}

enum ExtentKind {
    Semantic,
    Base(Vec<usize>), // tracks the (ordered) split factors
    Split {
        level: usize,
        factor: usize, // tracks the constant split factor
    }
}

// reference to a particular dimension of a buffer
struct DimRef<B> {
    buffer: B,
    dim: usize,
}

struct Axis(usize);

struct Input(usize);
enum Operand {
    Left,
    Right,
}
enum Arg {
    Readonly,
    Writeable,
}
struct Param {
    arg: Arg,
    ind: usize,
}

struct Shape<B>(Vec<Extent<B>>);

/*

- figure out semantic shape for every buffer
  - 

- every stage has a semantic shape

     o
    / \
   o   o
  / \ / \
 x  x x  x

*/

/*

- you can fuse with just node-local extents
  - instead of extents pointing to dim refs, just have them point to "axes" and
    resolve the axes later
    - then, fusion candidacy is just "are these loops of the same axis (mapped
      for the producer)?"

- compute expr-local node shape and loop extents

- in order to compute one "stage", we need:
  - alloc(s)
    - 
  - kernel dispatch
  - free(s)

- kernel
  - loops
    - extents (in terms of kernel inputs)
  - statement
    - op
    - accesses

*/

/*
notes on naming things
---
- Input is a program level binding
- Operand is expr level operand
- Arg is kernel level bucket (readonly/writeable)
- Param is a particular buffer passed to a kernel, e.g., readonly[0]

This means a Shape can be represented in 3 ways:
- `Shape<Operand>` the mapping of an i expression's operands to it's resulting
  shape. This is already canonicalized according to early-preference. This is
  derived from the semantics of the i expression. This does not directly appear
  in the generated code.
- `Shape<Input>` is the shape of one i component output in terms of the
  component inputs. This is canonicalized according to early-preference. This
  shape ultimately informs buffer layouts and the exported shape function.
- `Shape<Param>` is the shape of a kernel param. This is used  // TODO is this actually right??

*/


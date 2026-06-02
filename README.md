𝚒 is an experimental tensor computation language with declarative semantics,
explicit scheduling, and an extremely tiny surface area.

𝚒 aims to pull _just enough_ scheduling capability from the kernel DSL layer
into an otherwise high-level tensor language. The 𝚒 scheduling model only has
three concepts: loop splits, loop ordering, and input producer staging, but
these have predictable lowering consequences including loop/operator fusion,
storage folding, and online reduction rewriting.

These are all the necessary ingredients for implementing numerically stable
online blockwise [FlashAttention](https://arxiv.org/abs/2205.14135). Here is
what that looks like (it's dense but details of the syntax are given below):

```python
# does a matmul with right-hand input transposed
mm_t = i("ik*jk~ijk | i:16,j:16 | jii'j'k") >> i("+ijk~ij | i:16,j:16 | jii'j'k0")

# shifts values down by row-max for numerical stability
row_max_shift = (I & i(">ij~i | i:16,j:16 | ji0i'j'")) >> i("ij-i~ij | i:16,j:16 | ji01i'j'")

# applies exponentiation (first part of the softmax)
exp = i("^ij~ij | i:16,j:16 | ji0i'j'")

# normalizes along rows (second part of the softmax)
row_normalize = (I & i("+ij~i | i:16,j:16 | ji0i'j'")) >> i("ij/i~ij | i:16,j:16 | ji01i'j'")

# does a regular matmul
mm = i("ij*jk~ikj | i:16,j:16 | ji0i'kj'") >> i("+ikj~ik | i:16,j:16 | jii'kj'0")

# puts it all together
attn = mm_t >> row_max_shift >> exp >> row_normalize >> mm
```

# Thesis

The first "bet" of 𝚒 as a project was that a simple scheduling model could
admit FlashAttention-like target implementations. This bet has paid off,
although there is still the risk that things get messy as the language expands
to express a broader set of tensor computations.

The bet _now_ is that this simple scheduling model will make schedule _search_
more tractable. A lot of tensor compilers do search, but they do it in a complex
IR with too much configuration complexity. 𝚒 deliberately has fewer knobs to
turn. The bet is that they are the _right_ knobs. The scheduling model being
resident in the language (instead of an IR layer) means search won't happen
somewhere deep within the 𝚒 compiler, but over 𝚒 components. You write (or
trace from a Torch model) an 𝚒 component, and search simply finds you a better
one.

# Status

This project is at the proof-of-concept stage. There are significant gaps in the
language, the generated code is not yet performant, and the repo carries a lot
of AI agent debt.

Right now, we have [Python frontend](ilang-python) -> [runtime](core) ->
[compiler](compiler) -> [C backend](compiler/src/backends/c) working on Linux
and macOS, demonstrating the scheduling model, and allowing correctness
verification against NumPy/etc.

The 𝚒 compiler has no dependencies and generates a standalone dynamic library.
The 𝚒 runtime depends only on the compiler for the target platform.

# Running the FlashAttention [demo](ilang-python/flash-attn.py)

You will need Rust, Python, and NumPy installed.

```bash
cargo build --package i-core
python ilang-python/flash-attn.py
```

This computes a reference tensor with NumPy and then computes the same tensor
in 𝚒, once with the naive schedule and then again with the FlashAttention
schedule. The generated C code for FlashAttention is printed out for
inspection. We then assert the 𝚒-computed values match the NumPy reference to a
reasonable tolerance.

There is an annotated version of the C output
[here](ilang-python/annotated-flashattn.c).

# Language

𝚒 is a pure expression language. The base construct is the 𝚒 expression which
contains a scalar operation, indexing semantics, and scheduling information.
These are wired up into arbitrarily complex computation graphs using a small set
of combinators.

## 𝚒 expressions

In general, 𝚒 expressions are written with three "segments" delimited by `|`.
The first is the semantic expression, the second is a list of loop splits, and
the third is the schedule of loops and input staging directives.

The semantic part of 𝚒 expressions looks similar to einsum notation but does
not perform implicit summation. All reductions happen in their own expressions.
𝚒 expressions have a unary form (e.g. `-i~i`) and binary form (e.g. `i-i~i`).
Repeated input indices constrain the input shapes. For example, `i-i~i`
requires the left and right inputs be the same length. The shapes of the input
dimensions inform the shapes of the output. For example, the output of `i-i~i`
will be the same length as the inputs. Reductions are written by omitting input
indices from the output. For example, `+ij~i` performs a sum across the second
dimension of the input.

## Ops

| symbol | name  | default | reducible |
| ------ | ----- | ------- | --------- |
| `+`    | `add` | 0       | ✓         |
| `*`    | `mul` | 1       | ✓         |
| `-`    | `sub` | 0       |           |
| `/`    | `div` | 1       |           |
| `>`    | `max` | -∞      | ✓         |
| `<`    | `min` | ∞       | ✓         |
| `^`    | `pow` | e       |           |
| `$`    | `log` | e       |           |

The unary forms of 𝚒 expressions are just the binary forms with the default
value assumed on the left-hand side. The default value is chosen to be the
reduction identity if there is one, otherwise a value that gives sane unary
behavior. This way, we get `sub` -> `neg`, `div` -> `recip`, `pow` -> `exp`,
`log` -> `ln`.

## Combinators

The following combinators are used to compose 𝚒 expressions into computation
graphs called 𝚒 _components_.

| symbol | name    | semantics                       |
| ------ | ------- | ------------------------------- |
| `<<`   | compose | `(f << g)(x) = f(g(x))`         |
| `>>`   | chain   | `(f >> g)(x) = g(f(x))`         |
| `&`    | fanout  | `(f & g)(x) = (f(x), g(x))`     |
| `\|`   | pair    | `(f \| g)(x, y) = (f(x), g(y))` |
| `~`    | swap    | `(~f)(x, y) = f(y, x)`          |

## The 𝚒 scheduling model

The only scheduling concepts in 𝚒 are loop splits, loop ordering, and input
producer staging. Splits are declared in the second segment of an 𝚒 expression,
loops are ordered in the third segment, and inputs are staged within that same
loop ordering string.

Take for example: `+ijk~ij | i:16,k:16 | iki'jk'`. Here, the `i` and `k` axes
are each split by a factor of 16 (tiling each loop with a tile width of 16) and
the loops are ordered with the tile loops `i` and `k` on the outside and the
element loops `i'`, `j`, and `k'` on the inside.

If one 𝚒 expression (the _consumer_) takes another 𝚒 expression (the
_producer_) as input in the computation graph, the producer's computation can be
staged inside the schedule of the consumer. For example: `+ijk~ij | i:16,k:16 |
iki'jk'0` stages the 0-th input producer at the innermost loop of the consumer.

Staging producers in this way has three important lowering consequences:

1. If semantically equivalent consumer and producer loops above the stage site
are compatibly split and aligned, the 𝚒 compiler will _fuse_ them.
2. If the fusion allows one or more dimensions of an intermediate buffer to be
reused, it will _fold_ away these dimensions.
3. And finally, if a reduction is staged under a dependent reduction over the
same semantic axis, the reduction will be lowered into an online-corrected form
(the caveat is that this only works for supported reduction pairs, but once
supported, they are composable).

This scheduling model is the key difference between 𝚒 and other tensor
compilers. These lowering decisions are [_predictable_ consequences of the
model](https://pharr.org/matt/blog/2018/04/18/ispc-origins) rather than being
opaque compiler optimizations buried in some complex IR somewhere.

# Priorities

The critical priorities to take 𝚒 from proof-of-concept to being legitimately
useful on real-world workloads are the following:

1. **Language expressivity:** While 𝚒 can already express non-trivial tensor
computations like MLPs and attention mechanisms, it still has real gaps
including affine indexing, gather/scatter, prefix scan, logical/boolean ops,
and non-`f32` dtypes. Convolution is probably the most critical usability gap
at present. The goal is to add language features in as principled a way as
possible to keep 𝚒 small. We favor general and composable constructs over
purpose-specific ones.
2. **Backend maturity:** With the scheduling model demonstrated, most of the
performance will come from having backends that target fast hardware (i.e.
GPUs) and actually write _good_ code for them. Eventually hardware-specific
intrinsics will likely percolate up to the language layer, but only in such a
way that the semantics and scheduling do not depend on them.
3. **Interoperability:** If anyone is to actually use 𝚒, it needs to be made
really easy to do so. This means interfacing with existing infrastructure like
Torch.

# Inspiration

- [FlashAttention](https://arxiv.org/pdf/2205.14135)
- [Halide](https://people.csail.mit.edu/jrk/halide-pldi13.pdf)
- [TensorComprehensions](https://arxiv.org/pdf/1802.04730)
- [Torch einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- [tinygrad](https://github.com/tinygrad/tinygrad)

# Contact
Interested in 𝚒, tensor compilers, or related work? Reach out at
[contact@ilang.dev](mailto:contact@ilang.dev).

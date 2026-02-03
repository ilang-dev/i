𝚒 is a peculiar deep learning framework in the making. The thing that makes 𝚒
different from other frameworks is that it does not have primitive tensor ops.
Instead, it uses "index expressions" — a simple but powerful language for
applying scalar operations over multidimensional domains. Index expressions,
or 𝚒-expressions, are the atomic computation units which are then combined
into computation graphs using a small set of combinators.

The motivation for this peculiar language is two-fold. First, there is a
certain aesthetic appeal to creating a sufficiently expressive framework from a
small set of general components. And second, this description of computation is
particularly amenable to important scheduling optimizations like fusion and
tiling.

𝚒-expressions
---

𝚒-expressions look similar to einsum notation but do not perform implicit
summation. They apply a _single_ scalar operation over a multidimensional
domain defined by single-character indices. This forms a simple tensor
operation, either pointwise or reduction or reshape. These expressions are then
combined to form more complex operations. Here is a matrix multiply
implemented with 𝚒's Python front-end:

```python
i("ik*kj~ijk") >> i("+ijk~ij")
```

It is comprised of a "partial matrix product" expression chained into a "layer
accumulate" expression.

Breakdown of `ik*kj~ijk`:
- `ik`: indices of the left input, 2 chars => 2-dimensional
- `*`: applying scalar multiplication
- `kj`: indices of the right input, repeated `k` enforces a shape constraint;
  the 1 dimension of the left input corresponds to the same iteration domain as
  the 0 dimension of the right input, the familiar shape constraint of matrix
  multiplication.
- `~` syntax separating inputs from outputs
- `ijk` indices of the output, 3
  chars => 3-dimensional, output shape inferable from input shapes
- the iterative domain of this expression is given by `(i,j,k)`

In Python, this expression corresponds to something like:

``` python
for i in range(in0.shape[0]):
    for j in range(in1.shape[1]):
        for k in range(in0.shape[1]):
            out[i,j,k] = in0[i,k] * in1[k,j]
```

The second expression, `+ijk~ij`, is a reduction over the `k` dimension,
indicated by `k` appearing to the left of the `~` but not to the right.

ops
---

Here is the full list of ops.

| category | symbol | name  | default | reducible | implemented |
| -------- | ------ | ----- | ------- | --------- | ----------- |
| numeric  | `+`    | `add` | 0       | ✓         | ✓           |
|          | `*`    | `mul` | 1       | ✓         | ✓           |
|          | `-`    | `sub` | 0       |           | ✓           |
|          | `/`    | `div` | 1       |           | ✓           |
|          | `>`    | `max` | -inf    | ✓         | ✓           |
|          | `<`    | `min` | inf     | ✓         | ✓           |
|          | `^`    | `pow` | e       |           | ✓           |
|          | `$`    | `log` | e       |           | ✓           |
| boolean  | `>>`   | `gt`  | 0       |           |             |
|          | `>=`   | `gte` | 0       |           |             |
|          | `<<`   | `lt`  | 0       |           |             |
|          | `<=`   | `lte` | 0       |           |             |
|          | `==`   | `eq`  | 0       |           |             |
|          | `!=`   | `neq` | 0       |           |             |
| logical  | `&&`   | `and` | 1       | ✓         |             |
|          | `\|\|` | `or`  | 0       | ✓         |             |
|          | `^^`   | `xor` | 0       | ✓         |             |
|          | `!!`   | `not` | -       |           |             |

𝚒-expression forms
---

With the exception of `!!` (`not`), all 𝚒 ops can be written in pointwise
binary form: `i☐i~i`. This includes `^` (`pow`) and `$` (`log`) where the
left-hand-side is the base. All ops can also be written in pointwise unary
form: `☐i~i`. These can seen as the binary form, but where the left-hand-side
is taken to be the op's default value (given in the op table above). For
example, `-i~i` can be interpretted as `0-i~i`. Finally, ops which are
associative and have an identity value can be used as reductions, e.g.,
`+ij~i`.

All reducible ops have default value equal to their identity. Non-reducible ops
have a default value chosen to result in sane unary behavior. For example,
`pow` and `log` have default value `e` so that `pow(base, x)` becomes `exp(x)`
and `log(base, x)` becomes `ln(x)`.

Here are the various forms that 𝚒-expression can take including pointwise
binary, pointwise unary, and reduction. Some ops are not valid in all forms and
are omitted. Other ops are valid but are effectively no-ops. These are
indicated as such, but left in for completeness.

| form              | expr       | psuedo tensor notation |
| ----------------- | ---------- | ---------------------- |
| pointwise binary  | `i+i~i`    | `X+Y`                  |
|                   | `i*i~i`    | `X*Y`                  |
|                   | `i-i~i`    | `X-Y`                  |
|                   | `i/i~i`    | `X/Y`                  |
|                   | `i>i~i`    | `max(X,Y)`             |
|                   | `i<i~i`    | `min(X,Y)`             |
|                   | `i^i~i`    | `pow(base,X)`          |
|                   | `i$i~i`    | `log(base,X)`          |
|                   | `i>>i~i`   | `X>Y`                  |
|                   | `i>=i~i`   | `X>=Y`                 |
|                   | `i<<i~i`   | `X<Y`                  |
|                   | `i<=i~i`   | `X<=Y`                 |
|                   | `i==i~i`   | `X==Y`                 |
|                   | `i!=i~i`   | `X!=Y`                 |
|                   | `i&&i~i`   | `X&Y`                  |
|                   | `i\|\|i~i` | `X\|Y`                 |
|                   | `i^^i~i`   | `X^Y`                  |
| pointwise unary   | `+i~i`     | `X` (no-op)            |
|                   | `*i~i`     | `X` (no-op)            |
|                   | `-i~i`     | `-X`                   |
|                   | `/i~i`     | `1/X`                  |
|                   | `>i~i`     | `max(0,X)`             |
|                   | `<i~i`     | `min(0,X)`             |
|                   | `^i~i`     | `exp(X)`               |
|                   | `$i~i`     | `log(X)`               |
|                   | `>>i~i`    | `X>0`                  |
|                   | `>=i~i`    | `X>=0`                 |
|                   | `<<i~i`    | `X<0`                  |
|                   | `<=i~i`    | `X<=0`                 |
|                   | `==i~i`    | `X==0`                 |
|                   | `!=i~i`    | `X!=0`                 |
|                   | `&&i~i`    | `X&1` (no-op)          |
|                   | `\|\|i~i`  | `X\|0` (no-op)         |
|                   | `^^i~i`    | `X^0` (no-op)          |
|                   | `!!i~i`    | `!X`                   |
| reduction         | `+ij~i`    | `X.sum(dim=0)`         |
|                   | `*ij~i`    | `X.prod(dim=0)`        |
|                   | `>ij~i`    | `X.max(dim=0)`         |
|                   | `<ij~i`    | `X.min(dim=0)`         |
|                   | `&&ij~i`   | `X.all(dim=0)`         |
|                   | `\|\|ij~i` | `X.any(dim=0)`         |
|                   | `^^ij~i`   | `X.xor_reduce(dim=0)`  |

combinators
---

𝚒 programs are represented as computation graphs, leaves being the inputs,
roots being the outputs, and interior nodes being the intermediate results of
the computation. Individual 𝚒-expression are treated as atomic computation
graphs comprised of a single root and 1 or 2 leaves. The following combinators
are the graph wiring tools that allow for the composition of 𝚒-expressions into
arbitrarily complex computation graphs.

`compose` wires the leaves of one graph to the roots of another. `chain` does
the inverse, wiring the roots of one graph to the leaves of another. `fanout`
merges the inputs of two graphs, intuitively performing two different
computations on the same set of inputs. `pair` concatenates the roots and
leaves of two graphs such that they together can be handled as a single graph.
`swap` swaps the order of the first two roots of a single graph.

All of the combinators are well-defined for mixed arity. For example, `fanout`
merges inputs pairwise left-to-right and simply leaves any unpaired inputs
unmerged. Similarly, a mismatch between leaves and roots in compose leaves
the unpaired leaves or roots in the graph.

| symbol | name      | implemented |
| ------ | --------- | ----------- |
| `<<`   | `compose` | ✓           |
| `>>`   | `chain`   | ✓           |
| `&`    | `fanout`  | ✓           |
| `\|`   | `pair`    | ✓           |
| `~`    | `swap`    | ✓           |

Here is an example using `fanout` and `chain` to create a program to normalize
over the rows of a 2D input.

```python
row_sum = i("+ij~i")
row_div = i("ij/i~ij")
id = i("ij~ij")
row_normalize = (id & row_sum) >> row_div
```

scheduling
---
𝚒-expressions are declarative by default but can be extended with 𝚒's powerful
scheduling primitives. These include:
1. loop splitting
2. loop reordering
3. loop fusion

NOTE
- language notes
  - character indices present in the output and one of the inputs but absent from
    the other input are broadcasted, e.g., in the expression `ij+i~ij`, the `i` is
    broadcasted _over_ `j`.
  - distinguish between core i and framework/front-end
  - implicit one-way boolean->numeric casting
- present limitations of i
  - no sparsity support (including affine indexing)
  - no scatter/gather
  - no support for scans / prefix operations (e.g. cumulative sum) or general
    recurrences, where an iteration depends on the result of a previous
    iteration
- not yet implemented
  - autograd
- random
  - i has no state and is used for describing pure multidimensional functions
    of infinite domain
  - i lowers the computation graph into a scheduled intermediate representation
    with schedule elements like loop tiling, loop reordering, fusion, storage
    folding. eventually this will include target-specific annotations like
    TensorCores and SIMD vectorization
  - i is designed to be portable across various backends. currently Rust and
    CUDA backends are supported but ROCm, PTX, Triton, WGSL are planned as
    well. additionally, i could be used as a compiler stack for custom hardware
  - i is declarative with algorithm and scheduling specification completely
    separate, inspired by Halide
  - eventually i schedules will be searchable leading to extremely powerful
    automated optimization over various metrics like wall-clock time (obvious
    applications) and power (useful for edge AI for example)

## Inspiration
- [FlashAttention](https://arxiv.org/pdf/2205.14135) (hmm, could a compiler
  learn/find FlashAttention?)
- [TensorComprehensions](https://arxiv.org/pdf/1802.04730) (wow, terse DSLs are
  cool af)
- [Torch einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html)
  (damn, even more terse than TCs)
- [Halide](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) (decouple alg
  description from scheduling, search for fast kernels)
- [tinygrad](https://github.com/tinygrad/tinygrad) (simple good, search for
  fast kernels)


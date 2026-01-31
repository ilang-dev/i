𝚒 is a peculiar deep learning framework in the making. The thing that makes 𝚒
different from other frameworks is that it does not have pimitive tensor ops.
Instead, it uses "index expressions" – a simple but powerful language for
applying scalar operations over multidimensional domains. Index expressions,
or 𝚒-expressions, are the atomic computation units which are then combined
into computation graphs using a small set of combinators.

The motivation for this peculiar language is two-fold. First, there is a
certain aesthetic appeal to creating a sufficiently expressive framework from a
small set of general components. And second, this description of computation is
particularly ammenable to important scheduling optimizations like fusion and
tiling.

𝚒-expressions
---

𝚒-expressions are similar to einsum notation but without the implicit
summation. Here is a matrix multiply implemented with 𝚒's Python front-end:

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
- `~` syntax separating inputs from outputs `ijk` indices of the output, 3
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

The "standard form" for 𝚒-expressions is binary: `i☐i~i`, but they become unary
in two important ways. The first we have already encountered: reductions. Any
associative op with an identity can be used as a reduction. `+ijk~ij` reduces
over the `k` dimension by initializing the output to the identity of `+` (`0`)
and then accumulating along `k`. The second way is that every op has a default
value which is implicitly broadcasted on the left-hand-side. For example, `-`
has default value `0`, so `-i~i` is implicit for `0-i~i` and behaves like
negation.

All reducible ops have default value equal to their identity. Non-reducible ops
have a default value chosen to result in sane unary behavior. For example,
`pow` and `log` have default value `e` so that `pow(base, x)` becomes `exp(x)`
and `log(base, x)` becomes `ln(x)`.

The only intrinsically unary op is `!!` (`not`).

scheduling
---
TODO

ops
---

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

Numeric ops have single-char symbols. Boolean and logical ops have double-char
symbols.

NOTE
implicit one-way boolean->numeric casting

𝚒-expression forms
---

| form               | expr       | psuedo tensor notation |
| ------------------ | ---------- | ---------------------- |
| pointwise binary   | `i+i~i`    | `X+Y`                  |
|                    | `i*i~i`    | `X*Y`                  |
|                    | `i-i~i`    | `X-Y`                  |
|                    | `i/i~i`    | `X/Y`                  |
|                    | `i>i~i`    | `max(X,Y)`             |
|                    | `i<i~i`    | `min(X,Y)`             |
|                    | `i^i~i`    | `pow(base,X)`          |
|                    | `i$i~i`    | `log(base,X)`          |
|                    | `i>>i~i`   | `X>Y`                  |
|                    | `i>=i~i`   | `X>=Y`                 |
|                    | `i<<i~i`   | `X<Y`                  |
|                    | `i<=i~i`   | `X<=Y`                 |
|                    | `i==i~i`   | `X==Y`                 |
|                    | `i!=i~i`   | `X!=Y`                 |
|                    | `i&&i~i`   | `X&Y`                  |
|                    | `i\|\|i~i` | `X\|Y`                 |
|                    | `i^^i~i`   | `X^Y`                  |
| non-reducing unary | `+i~i`     | `X` (no-op)            |
|                    | `*i~i`     | `X` (no-op)            |
|                    | `-i~i`     | `-X`                   |
|                    | `/i~i`     | `1/X`                  |
|                    | `>i~i`     | `max(0,X)`             |
|                    | `<i~i`     | `min(0,X)`             |
|                    | `^i~i`     | `exp(X)`               |
|                    | `$i~i`     | `log(X)`               |
|                    | `>>i~i`    | `X>0`                  |
|                    | `>=i~i`    | `X>=0`                 |
|                    | `<<i~i`    | `X<0`                  |
|                    | `<=i~i`    | `X<=0`                 |
|                    | `==i~i`    | `X==0`                 |
|                    | `!=i~i`    | `X!=0`                 |
|                    | `&&i~i`    | `X&1` (no-op)          |
|                    | `\|\|i~i`  | `X\|0` (no-op)         |
|                    | `^^i~i`    | `X^0` (no-op)          |
|                    | `!!i~i`    | `!X`                   |
| reduction          | `+ij~i`    | `X.sum(dim=0)`         |
|                    | `*ij~i`    | `X.prod(dim=0)`        |
|                    | `>ij~i`    | `X.max(dim=0)`         |
|                    | `<ij~i`    | `X.min(dim=0)`         |
|                    | `&&ij~i`   | `X.all(dim=0)`         |
|                    | `\|\|ij~i` | `X.any(dim=0)`         |
|                    | `^^ij~i`   | `X.xor_reduce(dim=0)`  |

combinators
---

| symbol | name      | implemented |
| ------ | --------- | ----------- |
| `<<`   | `compose` | ✓           |
| `>>`   | `chain`   | ✓           |
| `&`    | `fanout`  | ✓           |
| `\|`   | `pair`    | ✓           |
| `~`    | `swap`    | ✓           |

NOTES:
- language notes
  - character indices present in the output and one of the inputs but absent from
    the other input are broadcasted, e.g., in the expression `ij+i~ij`, the `i` is
    broadcasted _over_ `j`.
  - distinguish between core i and framework/front-end
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


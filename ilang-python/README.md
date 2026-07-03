Python front-end for 𝚒.

This package exposes 𝚒 components as Python objects. Components compile lazily,
execute on CPU or CUDA according to their inputs, and interoperate with
`i.Tensor`, Python scalar/list literals, NumPy `array`s, and Torch `tensor`s.

## Package-style API

```python
import ilang
```

Exports:

- `ilang.Component`
- `ilang.Tensor`
- `ilang.Device`
- `ilang.Bench`
- `ilang.i`

## Preferred "DSL-style" API

The package-exported object `i` acts as a callable "namespace" that enables a
more compact style of 𝚒 code. When called, it constructors a `Component`, but
it also re-exposes much of the same package-level API as attributes.

```python
from ilang import i

i("+i~.")   # <ilang.component.Component object at ...>
i.Tensor    # <class 'ilang.tensor.Tensor'>
i.Component # <class 'ilang.component.Component'>
i.Device    # <enum 'Device'>
i.I         # mirrors `ilang.Component.I`
```

## Devices

A `Device` dictates where data lives and where compuation will run.

```python
i.Device.CPU  # "cpu"
i.Device.CUDA # "cuda"
```

## Tensors

`Tensor`s are immutable multidimensional data arrays.

```python
x = i.Tensor([[1, 2], [3, 4]]) # standard construction does shape-inference on nested list
x = i.Tensor([1, 2, 3, 4], shape=(2, 2)) # flat data with shape also works
```

Values are stored as `float32`.

### Attributes

```python
x.shape  # tuple[int, ...]
x.device # i.Device.CPU or i.Device.CUDA
x.data   # list[float] (only available on CPU tensors)
```

### Methods

```python
Tensor.to(device i.Device) -> Tensor # gives a new tensor on specified device

# examples:
x.to(i.Device.CUDA)
x.to(i.Device.CPU)
x.to("cuda")
x.to("cpu")
```

## Components

`i(expr: str) -> Component` parses one 𝚒 expression.

```python
f = i("+ij~i") # row-sum
```

`i.I` is the identity component.

### Component combinators

Components are combined using combinators. Components are immutable, so
combinators each return a new component.

Method forms:

```python
f.compose(g) # wires outputs of the right component into inputs of the left component
f.chain(g)   # wires outputs of the left component into inputs of the right component
f.fanout(g)  # shares inputs pairwise between two components
f.pair(g)    # concatenates the inputs and outputs of two components
f.swap()     # swaps the first two outputs of one component
```

Operator forms:

```python
f << g # f.compose(g)
f >> g # f.chain(g)
f & g  # f.fanout(g)
f | g  # f.pair(g)
~f     # f.swap()
```

Example:

```python
matmul = i("ik*kj~ijk") >> i("+ijk~ij")
```

## Execution

```python
out = matmul.exec(x, y)
```

`Component.exec(*inputs: TensorLike, into=None)` executes one component.

where `TensorLike = Tensor | torch.Tensor | numpy.ndarray | nested Python
sequence`

Execution device is determined by the input devices. All inputs must use the
same device. NumPy/Torch inputs must be contiguous and of dtype `float32`.
Python scalars/lists get promoted to CPU `ilang.Tensor`.

### Output type inference

The output container type and device are inferred from the input. For example,
`f.exec(torch.tensor([...], device="cuda"))` returns an `torch.Tensor` with
`device="cuda"`. This can be overridden with `into=`: `f.exec(nparray,
into=i.Tensor)`.

In general, component execution returns `Tuple[Tensor]` but is automatically
unpacked for single output results.

### Shape inference

`Component.output_shapes(*inputs)` returns one shape tuple per output. Shapes
are computed from input shapes without executing any kernels.

## Benchmarking

```python
bench = f.bench([x], n_warmups=10, n_runs=100)
```

`bench` executes warmup runs, records timed runs, and returns `Bench`.

`Bench` fields:

```python
bench.mean       # datetime.timedelta
bench.std        # datetime.timedelta
bench.n_warmups  # int
bench.n_runs     # int
bench.runs       # list[datetime.timedelta]
```

`repr(bench)` prints a compact human-readable timing summary.

## Errors

Invalid programs, invalid input types, invalid dtypes, non-contiguous arrays,
device mismatches, and backend failures raise Python exceptions.

Execution requires all inputs to reside on one device. No implicit CPU/CUDA
input synchronization is performed by `exec`.

CUDA tensor data is not read by `repr` and is not exposed by `.data`. Copy to
CPU explicitly.

## Examples

Native tensor execution:

```python
from ilang import i

f = i.i("+ij~ij")
x = i.Tensor([[1, 2], [3, 4]])
y = f.exec(x)
```

CUDA tensor execution:

```python
x = i.Tensor([[1, 2], [3, 4]]).to("cuda")
y = f.exec(x)
z = y.to("cpu")
```

NumPy execution:

```python
import numpy as np

x = np.ones((2, 2), dtype=np.float32)
y = f.exec(x)
```

Torch CUDA execution:

```python
import torch

x = torch.ones((2, 2), dtype=torch.float32, device="cuda")
y = f.exec(x)
```

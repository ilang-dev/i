# ilang-python

Python front-end for 𝚒.

This package exposes 𝚒 components as Python objects. Components compile lazily, execute on CPU or CUDA according to their inputs, and interoperate with `ilang.Tensor`, NumPy arrays, and Torch tensors.

## Public module

```python
import ilang as i
```

Exports:

- `i.i`
- `i.I`
- `i.Component`
- `i.Tensor`
- `i.DEVICE`
- `i.Bench`

## Components

A component is a parsed 𝚒 program tree.

```python
f = i.i("ik*kj~ij")
```

`i.i(src: str) -> Component` parses one component from source text.

`i.I` is the identity component. It forwards one input to one output.

`Component(src: str)` constructs a component directly. Prefer `i.i(src)` for source components.

### Component combinators

Components are immutable. Every combinator returns a new component.

```python
f.compose(g)
f.chain(g)
f.fanout(g)
f.pair(g)
f.swap()
```

Operator forms:

```python
f << g   # f.compose(g)
f >> g   # f.chain(g)
f & g    # f.fanout(g)
f | g    # f.pair(g)
~f       # f.swap()
```

Semantics:

- `compose` wires outputs of the right component into inputs of the left component.
- `chain` wires outputs of the left component into inputs of the right component.
- `fanout` shares inputs pairwise between two components.
- `pair` concatenates the inputs and outputs of two components.
- `swap` swaps the first two outputs of one component.

A string operand is parsed as a component before combination.

```python
f = i.i("+ij~ij") >> "*ij~ij"
```

### Generated source

Private inspection helpers return generated C/CUDA source strings.

```python
f._code()
f._cuda_code()
```

These methods are diagnostic API.

## Tensors

`Tensor` is the native Python tensor type for 𝚒.

```python
x = i.Tensor([[1, 2], [3, 4]])
```

Construction accepts a scalar or rectangular nested Python lists. Values are stored as `float32`.

Explicit flat data and shape are accepted.

```python
x = i.Tensor([1, 2, 3, 4], shape=(2, 2))
```

### Tensor attributes

```python
x.shape   # tuple[int, ...]
x.device  # i.DEVICE.CPU or i.DEVICE.CUDA
```

### Tensor data

```python
x.data  # list[float]
```

`data` is defined only for CPU tensors. CUDA tensor data is not directly accessible.

```python
x.to(i.DEVICE.CPU).data
```

### Devices

```python
i.DEVICE.CPU
i.DEVICE.CUDA
```

String aliases are accepted where a device is required:

```python
"cpu"
"cuda"
"gpu"
```

### Tensor transfer

```python
x_cuda = x.to(i.DEVICE.CUDA)
x_cpu = x_cuda.to(i.DEVICE.CPU)
```

`to` returns a tensor. If the tensor is already on the requested device, the same object is returned. Otherwise a new tensor is allocated and copied.

CUDA tensor allocation and copy are implemented by a lazily generated CUDA tensor runtime library. The core library does not link CUDA directly.

## Execution

```python
y = f.exec(x)
```

`Component.exec(*inputs, into=None)` executes one component.

Execution device is determined by the input devices:

- All CPU inputs execute the CPU program.
- All CUDA inputs execute the CUDA program.
- Mixed-device inputs are invalid.

The component compiles lazily for the selected backend. CPU execution uses generated C. CUDA execution uses generated CUDA.

### Input types

Inputs must be homogeneous by result type unless `into` is supplied.

Supported input types:

- `ilang.Tensor`
- `numpy.ndarray`
- `torch.Tensor`

NumPy inputs:

- must be `float32`
- must be C-contiguous
- are CPU inputs

Torch inputs:

- must be `torch.float32`
- must be contiguous
- may be CPU or CUDA

`ilang.Tensor` inputs carry their own device.

### Output type inference

Without `into`, the output container type is inferred from the input container type.

```python
f.exec(i.Tensor(...))      # returns ilang.Tensor or tuple[ilang.Tensor, ...]
f.exec(np_array)           # returns numpy.ndarray or tuple[numpy.ndarray, ...]
f.exec(torch_tensor)       # returns torch.Tensor or tuple[torch.Tensor, ...]
```

A single-output component returns one object. A multi-output component returns a tuple in output order.

Output device follows execution device:

- CPU `ilang.Tensor` inputs produce CPU `ilang.Tensor` outputs.
- CUDA `ilang.Tensor` inputs produce CUDA `ilang.Tensor` outputs.
- CPU Torch inputs produce CPU Torch outputs.
- CUDA Torch inputs produce CUDA Torch outputs.
- NumPy outputs are CPU-only.

### Explicit output target

`into` selects the output container family.

```python
f.exec(x, into=i.Tensor)
f.exec(x, into="tensor")
f.exec(x, into="numpy")
f.exec(x, into="torch")
f.exec(x, into=np.ndarray)
f.exec(x, into=torch.Tensor)
```

Accepted tensor aliases are `"tensor"`, `"ilang"`, and `"i"`.

Accepted NumPy aliases are `"numpy"` and `"np"`.

CUDA execution with NumPy output is invalid.

### Shape metadata

```python
shapes = f.output_shapes(*inputs)
```

`output_shapes` returns one shape tuple per output. Shapes are computed from input shapes without executing kernels.

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

Invalid programs, invalid input types, invalid dtypes, non-contiguous arrays, device mismatches, and backend failures raise Python exceptions.

Execution requires all inputs to reside on one device. No implicit CPU/CUDA input synchronization is performed by `exec`.

CUDA tensor data is not read by `repr` and is not exposed by `.data`. Copy to CPU explicitly.

## Examples

Native tensor execution:

```python
import ilang as i

f = i.i("+ij~ij")
x = i.Tensor([[1, 2], [3, 4]])
y = f.exec(x)
```

CUDA tensor execution:

```python
x = i.Tensor([[1, 2], [3, 4]]).to(i.DEVICE.CUDA)
y = f.exec(x)
z = y.to(i.DEVICE.CPU)
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

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from itertools import count
from math import floor, log10, sqrt
from typing import Any, ClassVar

from . import ffi
from .inputs import _inputs
from .tensor import Device, Tensor, _OwnedOutputs

__all__ = ["Bench", "Component"]

_INPUT_FREE = 0
_INPUT_BOUND = 1
_BIND_IDS = count()


@dataclass
class Bench:
    mean: timedelta
    std: timedelta
    n_warmups: int
    n_runs: int
    runs: list[timedelta]

    def _human_time(self) -> str:
        if self.mean.total_seconds() == 0:
            mean_order = -6
        else:
            mean_order = floor(log10(self.mean.total_seconds()))
        if mean_order <= -6:
            scale = 9
            unit = "ns"
        elif mean_order <= -3:
            scale = 6
            unit = "μs"
        elif mean_order <= 0:
            scale = 3
            unit = "ms"
        else:
            scale = 0
            unit = "s"
        mean_str = f"{round(self.mean.total_seconds() * 10**scale)}"
        std_str = f"{round(self.std.total_seconds() * 10**scale)}"
        return f"{mean_str}±{std_str} {unit}"

    def __repr__(self) -> str:
        return f"{self._human_time()}, warmups = {self.n_warmups}, runs = {self.n_runs}"


class Component:
    I: ClassVar[Component]  # noqa: E741

    def __init__(
        self,
        expr: str | None = None,
        _ptr: ctypes.c_void_p | None = None,
        _bound_values: dict[int, Any] | None = None,
    ) -> None:
        if _ptr is None:
            if expr is None:
                raise TypeError("Component needs expression")
            _ptr = ffi._core.i_parse(expr.encode())
        self._ptr: ctypes.c_void_p | None = ffi._check_ptr(_ptr)
        self._bound_values: dict[int, Any] = dict(_bound_values or {})
        for input in self._inputs():
            if input.kind == _INPUT_BOUND and int(input.id) not in self._bound_values:
                raise RuntimeError(f"missing value for bound input {int(input.id)}")
        self._program: ctypes.c_void_p | None = None
        self._cuda_program: ctypes.c_void_p | None = None

    def __del__(self) -> None:
        program = getattr(self, "_program", None)
        cuda_program = getattr(self, "_cuda_program", None)
        ptr = getattr(self, "_ptr", None)
        if program:
            ffi._core.i_program_free(program)
            self._program = None
        if cuda_program:
            ffi._core.i_program_free(cuda_program)
            self._cuda_program = None
        if ptr:
            ffi._core.i_component_free(ptr)
            self._ptr = None

    def _input_count(self) -> int:
        out = ctypes.c_size_t()
        ffi._check(ffi._core.i_component_input_count(self._ptr, ctypes.byref(out)))  # type: ignore[arg-type]
        return int(out.value)

    def _output_count(self) -> int:
        out = ctypes.c_size_t()
        ffi._check(ffi._core.i_component_output_count(self._ptr, ctypes.byref(out)))  # type: ignore[arg-type]
        return int(out.value)

    def _inputs(self, count: int | None = None) -> tuple[ffi._CInput, ...]:
        if count is None:
            count = self._input_count()
        inputs = (ffi._CInput * count)()
        ffi._check(ffi._core.i_component_inputs(self._ptr, inputs))  # type: ignore[arg-type]
        return tuple(inputs[i] for i in range(count))

    def _free_input_count(self) -> int:
        return sum(1 for input in self._inputs() if input.kind == _INPUT_FREE)

    def _bin(self, other: Component | str, fn: Any) -> Component:
        if not isinstance(other, Component):
            other = Component(other)
        return Component(
            _ptr=ffi._check_ptr(fn(self._ptr, other._ptr)),  # type: ignore[arg-type]
            _bound_values=_merge_bound_values(self, other),
        )

    def chain(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_chain)

    def compose(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_compose)

    def fanout(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_fanout)

    def pair(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_pair)

    def swap(self) -> Component:
        return Component(
            _ptr=ffi._check_ptr(ffi._core.i_swap(self._ptr)),  # type: ignore[arg-type]
            _bound_values=self._bound_values,
        )

    def bind(self, *args: Any) -> Component:
        free_count = self._free_input_count()
        if len(args) > free_count:
            raise TypeError(
                f"too many bindings: got {len(args)}, component has {free_count} free input(s)"
            )
        if not any(arg is not None for arg in args):
            return self

        adapter_parts: list[Component] = []
        for value in args:
            if value is None:
                adapter_parts.append(Component.I)
                continue
            bind_id = next(_BIND_IDS)
            adapter_parts.append(
                Component(
                    _ptr=ffi._check_ptr(ffi._core.i_bind(bind_id)),
                    _bound_values={bind_id: value},
                )
            )

        return _pair_components(adapter_parts).chain(self)

    def __call__(self, *args: Any, into: Any = None) -> Any:
        if not args:
            return self.exec(into=into)

        if into is not None:
            raise TypeError("into= is only valid when executing with an empty call")

        result = self
        pending_bindings = []
        for arg in args:
            if isinstance(arg, Component):
                if pending_bindings:
                    result = result.bind(*pending_bindings)
                    pending_bindings = []
                result = result.compose(arg)
            else:
                pending_bindings.append(arg)

        if pending_bindings:
            result = result.bind(*pending_bindings)
        return result

    def __rshift__(self, other: Component | str) -> Component:
        return self.chain(other)

    def __lshift__(self, other: Component | str) -> Component:
        return self.compose(other)

    def __and__(self, other: Component | str) -> Component:
        return self.fanout(other)

    def __or__(self, other: Component | str) -> Component:
        return self.pair(other)

    def __invert__(self) -> Component:
        return self.swap()

    def _compile(self, device: Device = Device.CPU) -> ctypes.c_void_p:
        if device is Device.CPU:
            if self._program is None:
                self._program = ffi._check_ptr(
                    ffi._core.i_compile(self._ptr, device._as_ffi())  # type: ignore[arg-type]
                )
            return self._program

        if self._cuda_program is None:
            self._cuda_program = ffi._check_ptr(
                ffi._core.i_compile(self._ptr, device._as_ffi())  # type: ignore[arg-type]
            )
        return self._cuda_program

    def _code(self, device: Device | str = Device.CPU) -> str:
        device = Device.coerce(device)
        s = ffi._check_ptr(ffi._core.i_code(self._ptr, device._as_ffi()))  # type: ignore[arg-type]
        try:
            ptr = ctypes.cast(s, ctypes.c_char_p).value
            if ptr is None:
                raise RuntimeError("Failed to decode None pointer")
            return ptr.decode()
        finally:
            ffi._core.i_string_free(s)

    def _cuda_code(self) -> str:
        return self._code(Device.CUDA)

    def _physical_inputs(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        interface = self._inputs()
        free_count = sum(1 for input in interface if input.kind == _INPUT_FREE)
        if len(inputs) > free_count:
            raise TypeError(
                f"too many inputs: got {len(inputs)}, component has {free_count} free input(s)"
            )

        physical: list[Any] = []
        missing: list[int] = []
        next_free = 0
        for physical_index, input in enumerate(interface):
            if input.kind == _INPUT_FREE:
                if next_free < len(inputs):
                    physical.append(inputs[next_free])
                    next_free += 1
                else:
                    missing.append(physical_index)
                continue

            if input.kind == _INPUT_BOUND:
                bind_id = int(input.id)
                try:
                    physical.append(self._bound_values[bind_id])
                except KeyError as err:
                    raise RuntimeError(f"missing value for bound input {bind_id}") from err
                continue

            raise RuntimeError(f"unknown component input kind {input.kind}")

        if missing:
            raise TypeError(f"component is not fully bound; missing input(s) {missing}")
        return tuple(physical)

    def output_shapes(
        self, *inputs: Any, _program: ctypes.c_void_p | None = None, _physical: bool = False
    ) -> list[tuple[int, ...]]:
        if not _physical:
            inputs = self._physical_inputs(inputs)
        program = _program if _program is not None else self._compile()
        input_arr, _keepalive = _inputs(inputs)
        count = ffi._core.i_output_count(program)
        ranks = (ctypes.c_size_t * count)()
        ffi._check(ffi._core.i_output_ranks(program, ranks))
        shape_bufs: list[Any] = [(ctypes.c_size_t * ranks[i])() for i in range(count)]
        shape_ptrs = (ctypes.POINTER(ctypes.c_size_t) * count)(
            *(ctypes.cast(buf, ctypes.POINTER(ctypes.c_size_t)) for buf in shape_bufs)
        )
        ffi._check(
            ffi._core.i_output_shapes(program, input_arr, len(inputs), shape_ptrs)
        )
        return [
            tuple(buf[j] for j in range(ranks[i])) for i, buf in enumerate(shape_bufs)
        ]

    def exec(self, *inputs: Any, into: Any = None) -> Any:
        inputs = self._physical_inputs(inputs)
        target, device = _resolve_target(inputs, into)
        program = self._compile(device)
        if target == "tensor" and device is Device.CPU:
            return self._exec_owned(program, *inputs)
        return self._exec_allocated(program, target, device, *inputs)

    def _exec_owned(self, program: ctypes.c_void_p, *inputs: Any) -> Any:
        input_arr, _keepalive = _inputs(inputs)
        outputs = ffi._core.i_exec(program, input_arr, len(inputs))
        if outputs.count == 0:
            ffi._check(-1)
        owner = _OwnedOutputs(outputs)
        tensors: list[Tensor] = [
            Tensor._from_owned(owner, i) for i in range(outputs.count)
        ]
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)

    def _exec_allocated(
        self, program: ctypes.c_void_p, target: str, device: Device, *inputs: Any
    ) -> Any:
        shapes: list[tuple[int, ...]] = self.output_shapes(*inputs, _program=program, _physical=True)
        if target == "numpy":
            import numpy as np

            if device is not Device.CPU:
                raise TypeError("NumPy outputs only support CPU execution")
            outs: list[Any] = [np.empty(shape, dtype=np.float32) for shape in shapes]
        elif target == "torch":
            import torch

            torch_device = "cuda" if device is Device.CUDA else "cpu"
            outs = [
                torch.empty(shape, dtype=torch.float32, device=torch_device)
                for shape in shapes
            ]
        elif target == "tensor":
            outs = [Tensor._empty(shape, device) for shape in shapes]
        else:
            raise TypeError(f"unknown execution target {target!r}")

        outputs = outs if len(outs) != 1 else outs[0]
        return self._exec_into(program, outputs, *inputs)

    def _exec_into(self, program: ctypes.c_void_p, outputs: Any, *inputs: Any) -> Any:
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        input_arr, _keepalive = _inputs(inputs)
        out_views: list[ffi._CTensorMut] = []
        out_keepalive: list[tuple[Any, ...]] = []
        for out in outputs:
            view, keep = _output(out)
            out_views.append(view)
            out_keepalive.append(keep)
        output_arr: ctypes.Array[ffi._CTensorMut] = (ffi._CTensorMut * len(out_views))(
            *out_views
        )
        ffi._check(
            ffi._core.i_exec_into(
                program, input_arr, len(inputs), output_arr, len(out_views)
            )
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def bench(
        self,
        *inputs: Any,
        n_warmups: int = 10,
        n_runs: int = 100,
    ) -> Bench:
        component = self.bind(*inputs) if inputs else self

        for _ in range(n_warmups):
            component.exec()

        runs = []
        for _ in range(n_runs):
            start = perf_counter()
            component.exec()
            end = perf_counter()
            runs.append(timedelta(seconds=end - start))

        mean = timedelta(seconds=sum([run.total_seconds() for run in runs]) / len(runs))
        std = timedelta(
            seconds=sqrt(
                1
                / (len(runs) - 1)
                * sum([(r - mean).total_seconds() ** 2 for r in runs])
            )
        )

        return Bench(
            mean=mean,
            std=std,
            n_warmups=n_warmups,
            n_runs=n_runs,
            runs=runs,
        )


def _merge_bound_values(*components: Component) -> dict[int, Any]:
    values: dict[int, Any] = {}
    for component in components:
        values.update(component._bound_values)
    return values


def _pair_components(components: list[Component]) -> Component:
    result = components[0]
    for component in components[1:]:
        result = result.pair(component)
    return result


def _resolve_target(inputs: tuple[Any, ...], into: Any) -> tuple[str, Device]:
    infos = [_input_info(x) for x in inputs]
    devices = {device for _kind, device in infos}
    if len(devices) != 1:
        if not devices:
            raise TypeError("cannot infer execution device without inputs")
        raise TypeError("all inputs must be on the same device")
    device = devices.pop()

    if into is not None:
        return _target_from_marker(into), device

    kinds = {kind for kind, _device in infos}
    if len(kinds) == 1:
        return kinds.pop(), device
    if not kinds:
        raise TypeError("cannot infer execution target without inputs; pass into=...")
    raise TypeError("cannot infer execution target from mixed input tensor types")


def _target_from_marker(marker: Any) -> str:
    if marker is Tensor:
        return "tensor"

    if isinstance(marker, str):
        name = marker.lower()
        if name in {"tensor", "ilang", "i"}:
            return "tensor"
        if name in {"numpy", "np"}:
            return "numpy"
        if name == "torch":
            return "torch"

    try:
        import numpy as np

        if marker is np.ndarray:
            return "numpy"
    except ImportError:
        pass

    try:
        import torch

        if marker is torch.Tensor:
            return "torch"
    except ImportError:
        pass

    raise TypeError(
        "into must be i.Tensor, 'numpy', 'torch', np.ndarray, or torch.Tensor"
    )


def _input_info(x: Any) -> tuple[str, Device]:
    if isinstance(x, Tensor):
        return "tensor", x.device

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                raise TypeError("NumPy inputs must be float32 and C-contiguous")
            return "numpy", Device.CPU
    except ImportError:
        pass

    try:
        import torch

        if isinstance(x, torch.Tensor):
            if str(x.dtype) != "torch.float32":
                raise TypeError("Torch tensors must be float32")
            if not x.is_contiguous():
                raise TypeError("Torch tensors must be contiguous")
            return "torch", Device.CUDA if x.is_cuda else Device.CPU
    except ImportError:
        pass

    try:
        Tensor(x)
    except (TypeError, ValueError):
        pass
    else:
        return "tensor", Device.CPU

    raise TypeError(
        "inputs must be ilang.Tensor, NumPy arrays, Torch tensors, "
        "or Python scalars/lists"
    )


def _output(x: Any) -> tuple[ffi._CTensorMut, tuple[Any, ...]]:
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                raise TypeError("NumPy outputs must be float32 and C-contiguous")
            shape, shape_buf = _shape_array(x.shape)
            data = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return ffi._CTensorMut(data, shape_buf, len(shape)), (x, shape_buf)
    except ImportError:
        pass

    try:
        import torch

        if isinstance(x, torch.Tensor):
            if str(x.dtype) != "torch.float32":
                raise TypeError("Torch outputs must be float32")
            if not x.is_contiguous():
                raise TypeError("Torch outputs must be contiguous")
            shape, shape_buf = _shape_array(tuple(x.shape))
            data = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
            return ffi._CTensorMut(data, shape_buf, len(shape)), (x, shape_buf)
    except ImportError:
        pass

    if isinstance(x, Tensor):
        return ffi._CTensorMut(x._data, x._shape_buf, len(x.shape)), (x,)

    raise TypeError("outputs must be ilang.Tensor, NumPy arrays, or Torch tensors")


def _shape_array(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], Any]:
    shape = tuple(int(d) for d in shape)
    arr: Any = (ctypes.c_size_t * len(shape))(*shape)
    return shape, arr


Component.I = Component(_ptr=ffi._core.i_identity())  # noqa: E741


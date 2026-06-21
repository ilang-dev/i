from __future__ import annotations

import ctypes
from typing import Any

from . import ffi
from .inputs import _inputs
from .tensor import Tensor, _OwnedOutputs

__all__ = ["Component", "I", "i"]


class Component:
    def __init__(
        self, src: str | None = None, _ptr: ctypes.c_void_p | None = None
    ) -> None:
        if _ptr is None:
            if src is None:
                raise TypeError("Component needs source")
            _ptr = ffi._core.i_parse(src.encode())
        self._ptr: ctypes.c_void_p | None = ffi._check_ptr(_ptr)
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

    def _bin(self, other: Component | str, fn: Any) -> Component:
        if not isinstance(other, Component):
            other = Component(other)
        return Component(_ptr=ffi._check_ptr(fn(self._ptr, other._ptr)))  # type: ignore[arg-type]

    def chain(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_chain)

    def compose(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_compose)

    def fanout(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_fanout)

    def pair(self, other: Component | str) -> Component:
        return self._bin(other, ffi._core.i_pair)

    def swap(self) -> Component:
        return Component(_ptr=ffi._check_ptr(ffi._core.i_swap(self._ptr)))  # type: ignore[return-value]

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

    def _compile(self) -> ctypes.c_void_p:
        if self._program is None:
            self._program = ffi._check_ptr(ffi._core.i_compile(self._ptr))  # type: ignore[arg-type]
        return self._program

    def _cuda_compile(self) -> ctypes.c_void_p:
        if self._cuda_program is None:
            self._cuda_program = ffi._check_ptr(
                ffi._core.i_cuda_compile(self._ptr),  # type: ignore[arg-type]
            )
        return self._cuda_program

    def _code(self) -> str:
        s = ffi._check_ptr(ffi._core.i_code(self._ptr))  # type: ignore[arg-type]
        try:
            ptr = ctypes.cast(s, ctypes.c_char_p).value
            if ptr is None:
                raise RuntimeError("Failed to decode None pointer")
            return ptr.decode()
        finally:
            ffi._core.i_string_free(s)

    def _cuda_code(self) -> str:
        s = ffi._check_ptr(ffi._core.i_cuda_code(self._ptr))  # type: ignore[arg-type]
        try:
            ptr = ctypes.cast(s, ctypes.c_char_p).value
            if ptr is None:
                raise RuntimeError("Failed to decode None pointer")
            return ptr.decode()
        finally:
            ffi._core.i_string_free(s)

    def output_shapes(self, *inputs: Any) -> list[tuple[int, ...]]:
        program = self._compile()
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

    def exec(self, *inputs: Any) -> Tensor | tuple[Tensor, ...]:
        program = self._compile()
        return self._exec_program(program, *inputs)

    def exec_cuda(self, *inputs: Any) -> Tensor | tuple[Tensor, ...]:
        program = self._cuda_compile()
        return self._exec_program(program, *inputs)

    def _exec_program(
        self, program: ctypes.c_void_p, *inputs: Any
    ) -> Tensor | tuple[Tensor, ...]:
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

    def exec_numpy(self, *inputs: Any) -> Any:
        import numpy as np

        shapes: list[tuple[int, ...]] = self.output_shapes(*inputs)
        outs: list[np.ndarray] = [np.empty(shape, dtype=np.float32) for shape in shapes]
        self.into(outs if len(outs) != 1 else outs[0], *inputs)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def exec_torch(self, *inputs: Any) -> Any:
        import torch

        shapes: list[tuple[int, ...]] = self.output_shapes(*inputs)
        outs: list[Any] = [
            torch.empty(shape, dtype=torch.float32, device="cpu") for shape in shapes
        ]
        self.into(outs if len(outs) != 1 else outs[0], *inputs)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def into(self, outputs: Any, *inputs: Any) -> Any:
        program = self._compile()
        return self._into_program(program, outputs, *inputs)

    def into_cuda(self, outputs: Any, *inputs: Any) -> Any:
        program = self._cuda_compile()
        return self._into_program(program, outputs, *inputs)

    def _into_program(
        self, program: ctypes.c_void_p, outputs: Any, *inputs: Any
    ) -> Any:
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

    if hasattr(x, "data_ptr") and hasattr(x, "shape"):
        if str(x.device) != "cpu":
            raise TypeError("Torch outputs must be on CPU")
        if str(x.dtype) != "torch.float32":
            raise TypeError("Torch outputs must be float32")
        if not x.is_contiguous():
            raise TypeError("Torch outputs must be contiguous")
        shape, shape_buf = _shape_array(tuple(x.shape))
        data = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        return ffi._CTensorMut(data, shape_buf, len(shape)), (x, shape_buf)

    raise TypeError("outputs must be NumPy arrays or Torch CPU tensors")


def _shape_array(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], Any]:
    shape = tuple(int(d) for d in shape)
    arr: Any = (ctypes.c_size_t * len(shape))(*shape)
    return shape, arr


I: Component = Component(_ptr=ffi._core.i_identity())  # noqa: E741


def i(src: str) -> Component:
    return Component(src)

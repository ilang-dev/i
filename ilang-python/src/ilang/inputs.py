from __future__ import annotations

import ctypes
from typing import Any

from . import ffi
from .tensor import Device, Tensor, _shape_array

__all__: list[str] = []


class _Input:
    def __init__(self, tensor: ffi._CTensor, keepalive: tuple[Any, ...]) -> None:
        self.tensor: ffi._CTensor = tensor
        self.keepalive: tuple[Any, ...] = keepalive


def _input(x: Any) -> _Input:
    if isinstance(x, Tensor):
        try:
            import torch

            if isinstance(x._owner, torch.Tensor):
                inner = _input(x._owner)
                return _Input(inner.tensor, (x, inner))
        except ImportError:
            pass
        return _Input(x._view(), (x,))

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                raise TypeError("NumPy inputs must be float32 and C-contiguous")
            shape, shape_buf = _shape_array(x.shape)
            data = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return _Input(ffi._CTensor(data, shape_buf, len(shape)), (x, shape_buf))
    except ImportError:
        pass

    try:
        import torch

        if isinstance(x, torch.Tensor):
            if str(x.dtype) != "torch.float32":
                raise TypeError("Torch tensors must be float32")
            if not x.is_contiguous():
                raise TypeError("Torch tensors must be contiguous")
            shape, shape_buf = _shape_array(tuple(x.shape))
            data = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
            return _Input(ffi._CTensor(data, shape_buf, len(shape)), (x, shape_buf))
    except ImportError:
        pass

    return _input(Tensor(x))


def _inputs(
    xs: list[Any] | tuple[Any, ...],
) -> tuple[ctypes.Array[ffi._CTensor], list[_Input]]:
    views: list[_Input] = [_input(x) for x in xs]
    arr: ctypes.Array[ffi._CTensor] = (ffi._CTensor * len(views))(
        *(v.tensor for v in views)
    )
    return arr, views

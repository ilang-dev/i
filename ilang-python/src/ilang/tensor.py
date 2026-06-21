from __future__ import annotations

import ctypes
from typing import Any

from . import ffi

__all__ = ["Tensor"]


def _shape_array(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], Any]:
    shape = tuple(int(d) for d in shape)
    arr: Any = (ctypes.c_size_t * len(shape))(*shape)
    return shape, arr


def _flatten(x: Any) -> tuple[tuple[int, ...], list[float]]:
    if isinstance(x, (int, float)):
        return (), [float(x)]
    if not isinstance(x, (list, tuple)):
        raise TypeError("Tensor expects a scalar or nested Python lists")
    if not x:
        return (0,), []

    child_shape, _data = _flatten(x[0])
    shape: tuple[int, ...] = (len(x),) + child_shape
    out: list[float] = []
    for item in x:
        item_shape, item_data = _flatten(item)
        if item_shape != child_shape:
            raise ValueError("ragged Tensor input")
        out.extend(item_data)
    return shape, out


class _OwnedOutputs:
    def __init__(self, outputs: ctypes.Structure) -> None:
        self.outputs: ctypes.Structure | None = outputs

    def __del__(self) -> None:
        outputs = getattr(self, "outputs", None)
        if outputs is not None:
            self.outputs = None
            ffi._core.i_outputs_free(outputs)


class Tensor:
    def __init__(self, x: Any, shape: tuple[int, ...] | None = None) -> None:
        if shape is None:
            shape, data = _flatten(x)
        else:
            shape = tuple(int(d) for d in shape)
            data = [float(v) for v in x]
        self.shape: tuple[int, ...] = tuple(shape)
        self._len: int = len(data)
        self._data: Any = (ctypes.c_float * self._len)(*data)
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner: _OwnedOutputs | None = None

    @classmethod
    def _from_owned(cls, owner: _OwnedOutputs, index: int) -> Tensor:
        outputs = owner.outputs
        assert outputs is not None
        raw = outputs.tensors[index]
        self: Tensor = cls.__new__(cls)
        self.shape = tuple(raw.shape[i] for i in range(raw.rank))
        self._len = raw.len
        self._data = raw.data
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner = owner
        return self

    @property
    def data(self) -> list[float]:
        return [self._data[i] for i in range(self._len)]

    def _view(self) -> ffi._CTensor:
        return ffi._CTensor(self._data, self._shape_buf, len(self.shape))

    def __del__(self) -> None:
        self._owner = None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data})"

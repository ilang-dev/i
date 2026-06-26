from __future__ import annotations

import ctypes
from enum import Enum
from typing import Any

from . import ffi

__all__ = ["DEVICE", "Tensor"]



class DEVICE(Enum):
    CPU = "cpu"
    CUDA = "cuda"

    @classmethod
    def coerce(cls, value: DEVICE | str) -> DEVICE:
        if isinstance(value, DEVICE):
            return value
        name = str(value).lower()
        if name in {"cpu", "device.cpu"}:
            return DEVICE.CPU
        if name in {"cuda", "gpu", "device.cuda"}:
            return DEVICE.CUDA
        raise ValueError(f"unknown device {value!r}")


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


def _torch_cuda_empty(shape: tuple[int, ...]) -> Any | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.empty(shape, dtype=torch.float32, device="cuda")
    except ImportError:
        pass
    return None


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


class _OwnedOutputs:
    def __init__(self, outputs: ctypes.Structure) -> None:
        self.outputs: ctypes.Structure | None = outputs

    def __del__(self) -> None:
        outputs = getattr(self, "outputs", None)
        if outputs is not None:
            self.outputs = None
            ffi._core.i_outputs_free(outputs)


class _CudaOwner:
    def __init__(self, data: Any) -> None:
        self.data: Any | None = data

    def __del__(self) -> None:
        data = getattr(self, "data", None)
        if data is not None:
            self.data = None
            ffi._core.i_cuda_free(data)


class Tensor:
    def __init__(
        self,
        x: Any,
        shape: tuple[int, ...] | None = None,
        *,
        device: DEVICE | str = DEVICE.CPU,
    ) -> None:
        device = DEVICE.coerce(device)
        if shape is None:
            shape, data = _flatten(x)
        else:
            shape = tuple(int(d) for d in shape)
            data = [float(v) for v in x]
        self.shape: tuple[int, ...] = tuple(shape)
        self.device: DEVICE = DEVICE.CPU
        self._len: int = len(data)
        self._data: Any = (ctypes.c_float * self._len)(*data)
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner: _OwnedOutputs | _CudaOwner | None = None
        if device is DEVICE.CUDA:
            moved = self.to(DEVICE.CUDA)
            self.device = moved.device
            self._data = moved._data
            self._owner = moved._owner
            moved._owner = None

    @classmethod
    def _from_owned(cls, owner: _OwnedOutputs, index: int) -> Tensor:
        outputs = owner.outputs
        assert outputs is not None
        raw = outputs.tensors[index]
        self: Tensor = cls.__new__(cls)
        self.shape = tuple(raw.shape[i] for i in range(raw.rank))
        self.device = DEVICE.CPU
        self._len = raw.len
        self._data = raw.data
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner = owner
        return self

    @classmethod
    def _empty(cls, shape: tuple[int, ...], device: DEVICE | str) -> Tensor:
        device = DEVICE.coerce(device)
        self: Tensor = cls.__new__(cls)
        self.shape = tuple(int(d) for d in shape)
        self.device = device
        self._len = _numel(self.shape)
        self._shape, self._shape_buf = _shape_array(self.shape)
        if device is DEVICE.CPU:
            self._data = (ctypes.c_float * self._len)()
            self._owner = None
        else:
            torch_owner = _torch_cuda_empty(self.shape)
            if torch_owner is not None:
                self._data = ctypes.cast(torch_owner.data_ptr(), ctypes.POINTER(ctypes.c_float))
                self._owner = torch_owner
            else:
                data = ffi._check_ptr(ffi._core.i_cuda_alloc(self._len))
                self._data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
                self._owner = _CudaOwner(self._data)
        return self

    @property
    def data(self) -> list[float]:
        if self.device is not DEVICE.CPU:
            raise RuntimeError(
                "CUDA tensor data is not directly accessible; call .to(DEVICE.CPU) first"
            )
        return [self._data[i] for i in range(self._len)]

    def to(self, device: DEVICE | str) -> Tensor:
        device = DEVICE.coerce(device)
        if device is self.device:
            return self
        out = Tensor._empty(self.shape, device)
        if self.device is DEVICE.CPU and device is DEVICE.CUDA:
            ffi._check(
                ffi._core.i_cuda_copy_from_host(out._data, self._data, self._len)
            )
        elif self.device is DEVICE.CUDA and device is DEVICE.CPU:
            ffi._check(ffi._core.i_cuda_copy_to_host(out._data, self._data, self._len))
        else:
            raise RuntimeError(f"unsupported tensor copy {self.device} -> {device}")
        return out

    def _view(self) -> ffi._CTensor:
        try:
            import torch

            if isinstance(self._owner, torch.Tensor):
                data = ctypes.cast(self._owner.data_ptr(), ctypes.POINTER(ctypes.c_float))
                return ffi._CTensor(data, self._shape_buf, len(self.shape))
        except ImportError:
            pass
        return ffi._CTensor(self._data, self._shape_buf, len(self.shape))

    def __del__(self) -> None:
        self._owner = None

    def __repr__(self) -> str:
        if self.device is DEVICE.CPU:
            return f"Tensor(shape={self.shape}, device=CPU, data={self.data})"
        return f"Tensor(shape={self.shape}, device=CUDA)"

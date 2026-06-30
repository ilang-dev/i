from __future__ import annotations

import ctypes
from enum import Enum
from numbers import Real
from typing import Any

from . import ffi

__all__ = ["Device", "Tensor"]



class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"

    def _as_ffi(self) -> int:
        return 0 if self is Device.CPU else 1

    @classmethod
    def coerce(cls, value: Device | str) -> Device:
        if isinstance(value, Device):
            return value
        name = str(value).lower()
        if name in {"cpu", "device.cpu"}:
            return Device.CPU
        if name in {"cuda", "gpu", "device.cuda"}:
            return Device.CUDA
        raise ValueError(f"unknown device {value!r}")


def _shape_array(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], Any]:
    shape = tuple(int(d) for d in shape)
    arr: Any = (ctypes.c_size_t * len(shape))(*shape)
    return shape, arr


def _flatten(x: Any) -> tuple[tuple[int, ...], list[float]]:
    if isinstance(x, Real):
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


class _DeviceOwner:
    def __init__(self, device: Device, data: Any) -> None:
        self.device = device
        self.data: Any | None = data

    def __del__(self) -> None:
        data = getattr(self, "data", None)
        if data is not None:
            self.data = None
            ffi._core.i_free(self.device._as_ffi(), data)


class Tensor:
    def __init__(
        self,
        x: Any,
        shape: tuple[int, ...] | None = None,
        *,
        device: Device | str = Device.CPU,
    ) -> None:
        device = Device.coerce(device)
        if shape is None:
            shape, data = _flatten(x)
        else:
            shape = tuple(int(d) for d in shape)
            data = [float(v) for v in x]
        self.shape: tuple[int, ...] = tuple(shape)
        self.device: Device = Device.CPU
        self._len: int = len(data)
        self._data: Any = (ctypes.c_float * self._len)(*data)
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner: _OwnedOutputs | _DeviceOwner | None = None
        if device is Device.CUDA:
            moved = self.to(Device.CUDA)
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
        self.device = Device.CPU
        self._len = raw.len
        self._data = raw.data
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner = owner
        return self

    @classmethod
    def _empty(cls, shape: tuple[int, ...], device: Device | str) -> Tensor:
        device = Device.coerce(device)
        self: Tensor = cls.__new__(cls)
        self.shape = tuple(int(d) for d in shape)
        self.device = device
        self._len = _numel(self.shape)
        self._shape, self._shape_buf = _shape_array(self.shape)
        if device is Device.CPU:
            self._data = (ctypes.c_float * self._len)()
            self._owner = None
        else:
            torch_owner = _torch_cuda_empty(self.shape)
            if torch_owner is not None:
                self._data = ctypes.cast(torch_owner.data_ptr(), ctypes.POINTER(ctypes.c_float))
                self._owner = torch_owner
            else:
                data = ffi._check_ptr(ffi._core.i_alloc(device._as_ffi(), self._len))
                self._data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
                self._owner = _DeviceOwner(device, self._data)
        return self

    @property
    def data(self) -> list[float]:
        if self.device is not Device.CPU:
            raise RuntimeError(
                "CUDA tensor data is not directly accessible; call .to(Device.CPU) first"
            )
        return [self._data[i] for i in range(self._len)]

    def to(self, device: Device | str) -> Tensor:
        device = Device.coerce(device)
        if device is self.device:
            return self
        out = Tensor._empty(self.shape, device)
        ffi._check(
            ffi._core.i_copy(
                out.device._as_ffi(),
                out._data,
                self.device._as_ffi(),
                self._data,
                self._len,
            )
        )
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
        if self.device is Device.CPU:
            return f"Tensor(shape={self.shape}, device=CPU, data={self.data})"
        return f"Tensor(shape={self.shape}, device=CUDA)"

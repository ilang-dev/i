from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _load_core() -> ctypes.CDLL:
    override = os.environ.get("I_CORE_LIB")
    if override:
        return ctypes.CDLL(override)

    here = Path(__file__).resolve()
    names: dict[str, list[str]] = {
        "darwin": ["libi_core.dylib"],
        "win32": ["i_core.dll"],
    }
    so_name: list[str] = names.get(sys.platform, ["libi_core.so"])

    roots: list[Path] = [
        here.parent,
        here.parent.parent / "target" / "release",
        here.parent.parent / "target" / "debug",
    ]
    for root in roots:
        for name in so_name:
            path = root / name
            if path.exists():
                return ctypes.CDLL(str(path))

    raise RuntimeError("could not find i-core library; run `cargo build -p i-core`")


_core: ctypes.CDLL = _load_core()


def _check_ptr(ptr: ctypes.c_void_p | None) -> ctypes.c_void_p:
    if not ptr:
        err = _core.i_error()
        raise RuntimeError(err.decode() if err else "i-core error")
    return ptr


def _check(code: int) -> None:
    if code != 0:
        err = _core.i_error()
        raise RuntimeError(err.decode() if err else "i-core error")


class _CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
    ]


class _CTensorMut(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
    ]


class _COwnedTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
        ("len", ctypes.c_size_t),
    ]


class _COutputs(ctypes.Structure):
    _fields_ = [
        ("tensors", ctypes.POINTER(_COwnedTensor)),
        ("count", ctypes.c_size_t),
    ]


def _bind_functions(core: ctypes.CDLL) -> None:
    core.i_parse.argtypes = [ctypes.c_char_p]
    core.i_parse.restype = ctypes.c_void_p
    core.i_identity.argtypes = []
    core.i_identity.restype = ctypes.c_void_p
    for _name in ("i_chain", "i_compose", "i_fanout", "i_pair"):
        _fn = getattr(core, _name)
        _fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _fn.restype = ctypes.c_void_p
    core.i_swap.argtypes = [ctypes.c_void_p]
    core.i_swap.restype = ctypes.c_void_p
    core.i_bind_input.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    core.i_bind_input.restype = ctypes.c_void_p
    core.i_component_input_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    core.i_component_input_count.restype = ctypes.c_int
    core.i_component_output_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    core.i_component_output_count.restype = ctypes.c_int
    core.i_component_input_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    core.i_component_input_states.restype = ctypes.c_int
    core.i_code.argtypes = [ctypes.c_void_p, ctypes.c_int]
    core.i_code.restype = ctypes.c_void_p
    core.i_compile.argtypes = [ctypes.c_void_p, ctypes.c_int]
    core.i_compile.restype = ctypes.c_void_p
    core.i_program_device.argtypes = [ctypes.c_void_p]
    core.i_program_device.restype = ctypes.c_int
    core.i_alloc.argtypes = [ctypes.c_int, ctypes.c_size_t]
    core.i_alloc.restype = ctypes.c_void_p
    core.i_free.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    core.i_copy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    core.i_copy.restype = ctypes.c_int
    core.i_output_count.argtypes = [ctypes.c_void_p]
    core.i_output_count.restype = ctypes.c_size_t
    core.i_output_ranks.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    core.i_output_ranks.restype = ctypes.c_int
    core.i_output_shapes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CTensor),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ]
    core.i_output_shapes.restype = ctypes.c_int
    core.i_exec_into.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CTensor),
        ctypes.c_size_t,
        ctypes.POINTER(_CTensorMut),
        ctypes.c_size_t,
    ]
    core.i_exec_into.restype = ctypes.c_int
    core.i_exec.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CTensor), ctypes.c_size_t]
    core.i_exec.restype = _COutputs
    core.i_component_free.argtypes = [ctypes.c_void_p]
    core.i_program_free.argtypes = [ctypes.c_void_p]
    core.i_outputs_free.argtypes = [_COutputs]
    core.i_string_free.argtypes = [ctypes.c_void_p]
    core.i_error.argtypes = []
    core.i_error.restype = ctypes.c_char_p


_core: ctypes.CDLL = _load_core()
_bind_functions(_core)

__all__ = ["_core", "_check_ptr", "_check", "_CTensor", "_CTensorMut"]

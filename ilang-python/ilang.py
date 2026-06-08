import ctypes
import os
from pathlib import Path


def _load_core():
    override = os.environ.get("ILANG_CORE_LIB")
    if override:
        return ctypes.CDLL(override)

    here = Path(__file__).resolve()
    names = {
        "darwin": ["libi_core.dylib"],
        "win32": ["i_core.dll"],
    }.get(os.sys.platform, ["libi_core.so"])

    roots = [
        here.parent,
        here.parent.parent / "target" / "release",
        here.parent.parent / "target" / "debug",
    ]
    for root in roots:
        for name in names:
            path = root / name
            if path.exists():
                return ctypes.CDLL(str(path))

    raise RuntimeError("could not find i-core library; run `cargo build -p i-core`")


_core = _load_core()


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


_core.i_parse.argtypes = [ctypes.c_char_p]
_core.i_parse.restype = ctypes.c_void_p
_core.i_identity.argtypes = []
_core.i_identity.restype = ctypes.c_void_p
for _name in ("i_chain", "i_compose", "i_fanout", "i_pair"):
    _fn = getattr(_core, _name)
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _fn.restype = ctypes.c_void_p
_core.i_swap.argtypes = [ctypes.c_void_p]
_core.i_swap.restype = ctypes.c_void_p
_core.i_code.argtypes = [ctypes.c_void_p]
_core.i_code.restype = ctypes.c_void_p
_core.i_compile.argtypes = [ctypes.c_void_p]
_core.i_compile.restype = ctypes.c_void_p
_core.i_output_count.argtypes = [ctypes.c_void_p]
_core.i_output_count.restype = ctypes.c_size_t
_core.i_output_ranks.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
_core.i_output_ranks.restype = ctypes.c_int
_core.i_output_shapes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CTensor),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
]
_core.i_output_shapes.restype = ctypes.c_int
_core.i_exec_into.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CTensor),
    ctypes.c_size_t,
    ctypes.POINTER(_CTensorMut),
    ctypes.c_size_t,
]
_core.i_exec_into.restype = ctypes.c_int
_core.i_exec.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CTensor), ctypes.c_size_t]
_core.i_exec.restype = _COutputs
_core.i_component_free.argtypes = [ctypes.c_void_p]
_core.i_program_free.argtypes = [ctypes.c_void_p]
_core.i_outputs_free.argtypes = [_COutputs]
_core.i_string_free.argtypes = [ctypes.c_void_p]
_core.i_error.argtypes = []
_core.i_error.restype = ctypes.c_char_p


def _check_ptr(ptr):
    if not ptr:
        err = _core.i_error()
        raise RuntimeError(err.decode() if err else "i-core error")
    return ptr


def _check(code):
    if code != 0:
        err = _core.i_error()
        raise RuntimeError(err.decode() if err else "i-core error")


def _shape_array(shape):
    shape = tuple(int(d) for d in shape)
    arr = (ctypes.c_size_t * len(shape))(*shape)
    return shape, arr


def _flatten(x):
    if isinstance(x, (int, float)):
        return (), [float(x)]
    if not isinstance(x, (list, tuple)):
        raise TypeError("Tensor expects a scalar or nested Python lists")
    if not x:
        return (0,), []

    child_shape, data = _flatten(x[0])
    shape = (len(x),) + child_shape
    out = []
    for item in x:
        item_shape, item_data = _flatten(item)
        if item_shape != child_shape:
            raise ValueError("ragged Tensor input")
        out.extend(item_data)
    return shape, out


class Tensor:
    def __init__(self, x, shape=None):
        if shape is None:
            shape, data = _flatten(x)
        else:
            shape = tuple(int(d) for d in shape)
            data = [float(v) for v in x]
        self.shape = tuple(shape)
        self._len = len(data)
        self._data = (ctypes.c_float * self._len)(*data)
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner = None

    @classmethod
    def _from_owned(cls, owner, index):
        raw = owner.outputs.tensors[index]
        self = cls.__new__(cls)
        self.shape = tuple(raw.shape[i] for i in range(raw.rank))
        self._len = raw.len
        self._data = raw.data
        self._shape, self._shape_buf = _shape_array(self.shape)
        self._owner = owner
        return self

    @property
    def data(self):
        return [self._data[i] for i in range(self._len)]

    def _view(self):
        return _CTensor(self._data, self._shape_buf, len(self.shape))

    def __del__(self):
        self._owner = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"


class _OwnedOutputs:
    def __init__(self, outputs):
        self.outputs = outputs

    def __del__(self):
        outputs = getattr(self, "outputs", None)
        if outputs is not None:
            self.outputs = None
            _core.i_outputs_free(outputs)


class _Input:
    def __init__(self, tensor, keepalive):
        self.tensor = tensor
        self.keepalive = keepalive


def _input(x):
    if isinstance(x, Tensor):
        return _Input(x._view(), x)

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.float32 or not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
            shape, shape_buf = _shape_array(arr.shape)
            data = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return _Input(_CTensor(data, shape_buf, len(shape)), (arr, shape_buf))
    except ImportError:
        pass

    if hasattr(x, "data_ptr") and hasattr(x, "shape"):
        if str(x.device) != "cpu":
            raise TypeError("Torch tensors must be on CPU")
        if str(x.dtype) != "torch.float32":
            raise TypeError("Torch tensors must be float32")
        if not x.is_contiguous():
            x = x.contiguous()
        shape, shape_buf = _shape_array(tuple(x.shape))
        data = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        return _Input(_CTensor(data, shape_buf, len(shape)), (x, shape_buf))

    return _input(Tensor(x))


def _inputs(xs):
    views = [_input(x) for x in xs]
    arr = (_CTensor * len(views))(*(v.tensor for v in views))
    return arr, views


class Component:
    def __init__(self, src=None, _ptr=None):
        if _ptr is None:
            if src is None:
                raise TypeError("Component needs source")
            _ptr = _core.i_parse(src.encode())
        self._ptr = _check_ptr(_ptr)
        self._program = None

    def __del__(self):
        program = getattr(self, "_program", None)
        ptr = getattr(self, "_ptr", None)
        if program:
            _core.i_program_free(program)
            self._program = None
        if ptr:
            _core.i_component_free(ptr)
            self._ptr = None

    def _bin(self, other, fn):
        if not isinstance(other, Component):
            other = Component(other)
        return Component(_ptr=_check_ptr(fn(self._ptr, other._ptr)))

    def chain(self, other):
        return self._bin(other, _core.i_chain)

    def compose(self, other):
        return self._bin(other, _core.i_compose)

    def fanout(self, other):
        return self._bin(other, _core.i_fanout)

    def pair(self, other):
        return self._bin(other, _core.i_pair)

    def swap(self):
        return Component(_ptr=_check_ptr(_core.i_swap(self._ptr)))

    def __rshift__(self, other):
        return self.chain(other)

    def __lshift__(self, other):
        return self.compose(other)

    def __and__(self, other):
        return self.fanout(other)

    def __or__(self, other):
        return self.pair(other)

    def __invert__(self):
        return self.swap()

    def _compile(self):
        if self._program is None:
            self._program = _check_ptr(_core.i_compile(self._ptr))
        return self._program

    def _code(self):
        s = _check_ptr(_core.i_code(self._ptr))
        try:
            return ctypes.cast(s, ctypes.c_char_p).value.decode()
        finally:
            _core.i_string_free(s)

    def output_shapes(self, *inputs):
        program = self._compile()
        input_arr, keepalive = _inputs(inputs)
        count = _core.i_output_count(program)
        ranks = (ctypes.c_size_t * count)()
        _check(_core.i_output_ranks(program, ranks))
        shape_bufs = [(ctypes.c_size_t * ranks[i])() for i in range(count)]
        shape_ptrs = (ctypes.POINTER(ctypes.c_size_t) * count)(
            *(ctypes.cast(buf, ctypes.POINTER(ctypes.c_size_t)) for buf in shape_bufs)
        )
        _check(_core.i_output_shapes(program, input_arr, len(inputs), shape_ptrs))
        return [tuple(buf[j] for j in range(ranks[i])) for i, buf in enumerate(shape_bufs)]

    def exec(self, *inputs):
        program = self._compile()
        input_arr, keepalive = _inputs(inputs)
        outputs = _core.i_exec(program, input_arr, len(inputs))
        if outputs.count == 0:
            _check(-1)
        owner = _OwnedOutputs(outputs)
        tensors = [Tensor._from_owned(owner, i) for i in range(outputs.count)]
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)

    def exec_numpy(self, *inputs):
        import numpy as np

        shapes = self.output_shapes(*inputs)
        outs = [np.empty(shape, dtype=np.float32) for shape in shapes]
        self.into(outs if len(outs) != 1 else outs[0], *inputs)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def exec_torch(self, *inputs):
        import torch

        shapes = self.output_shapes(*inputs)
        outs = [torch.empty(shape, dtype=torch.float32, device="cpu") for shape in shapes]
        self.into(outs if len(outs) != 1 else outs[0], *inputs)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def into(self, outputs, *inputs):
        program = self._compile()
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        input_arr, keepalive = _inputs(inputs)
        out_views = []
        out_keepalive = []
        for out in outputs:
            view, keep = _output(out)
            out_views.append(view)
            out_keepalive.append(keep)
        output_arr = (_CTensorMut * len(out_views))(*out_views)
        _check(_core.i_exec_into(program, input_arr, len(inputs), output_arr, len(out_views)))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _output(x):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.dtype != np.float32 or not x.flags.c_contiguous:
                raise TypeError("NumPy outputs must be float32 and C-contiguous")
            shape, shape_buf = _shape_array(x.shape)
            data = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return _CTensorMut(data, shape_buf, len(shape)), (x, shape_buf)
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
        return _CTensorMut(data, shape_buf, len(shape)), (x, shape_buf)

    raise TypeError("outputs must be NumPy arrays or Torch CPU tensors")


I = Component(_ptr=_core.i_identity())


def i(src):
    return Component(src)

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .. import ffi


@dataclass
class StoredTensor:
    tensor_id: str
    shape: tuple[int, ...]
    data: ctypes.POINTER(ctypes.c_float)
    shape_buf: Any

    @property
    def length(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def view(self) -> ffi._CTensor:
        return ffi._CTensor(self.data, self.shape_buf, len(self.shape))

    def mut_view(self) -> ffi._CTensorMut:
        return ffi._CTensorMut(self.data, self.shape_buf, len(self.shape))


class Program:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.count.argtypes = []
        self.lib.count.restype = ctypes.c_size_t
        self.lib.ranks.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        self.lib.ranks.restype = None
        self.lib.shapes.argtypes = [
            ctypes.POINTER(ffi._CTensor),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
        ]
        self.lib.shapes.restype = None
        self.lib.exec.argtypes = [
            ctypes.POINTER(ffi._CTensor),
            ctypes.POINTER(ffi._CTensorMut),
        ]
        self.lib.exec.restype = None

    def output_shapes(self, inputs: list[StoredTensor]) -> list[tuple[int, ...]]:
        count = int(self.lib.count())
        ranks = (ctypes.c_size_t * count)()
        self.lib.ranks(ranks)
        shape_bufs: list[Any] = [(ctypes.c_size_t * int(ranks[i]))() for i in range(count)]
        shape_ptrs = (ctypes.POINTER(ctypes.c_size_t) * count)(
            *(ctypes.cast(buf, ctypes.POINTER(ctypes.c_size_t)) for buf in shape_bufs)
        )
        input_arr = _input_array(inputs)
        self.lib.shapes(input_arr, shape_ptrs)
        return [tuple(int(buf[j]) for j in range(int(ranks[i]))) for i, buf in enumerate(shape_bufs)]

    def exec(self, inputs: list[StoredTensor], outputs: list[StoredTensor]) -> None:
        self.lib.exec(_input_array(inputs), _output_array(outputs))


def _shape_buf(shape: tuple[int, ...]) -> Any:
    return (ctypes.c_size_t * len(shape))(*shape)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def _input_array(inputs: list[StoredTensor]) -> Any:
    return (ffi._CTensor * len(inputs))(*(tensor.view() for tensor in inputs))


def _output_array(outputs: list[StoredTensor]) -> Any:
    return (ffi._CTensorMut * len(outputs))(*(tensor.mut_view() for tensor in outputs))


class State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tensors: dict[str, StoredTensor] = {}
        self.programs: dict[str, Program] = {}

    def alloc_tensor(self, shape: tuple[int, ...]) -> StoredTensor:
        length = _numel(shape)
        data = ffi._check_ptr(ffi._core.i_cuda_alloc(length))
        tensor = StoredTensor(
            tensor_id=uuid.uuid4().hex,
            shape=shape,
            data=ctypes.cast(data, ctypes.POINTER(ctypes.c_float)),
            shape_buf=_shape_buf(shape),
        )
        with self.lock:
            self.tensors[tensor.tensor_id] = tensor
        return tensor

    def upload_tensor(self, shape: tuple[int, ...], values: list[Any]) -> StoredTensor:
        length = _numel(shape)
        if len(values) != length:
            raise ValueError(f"data length {len(values)} does not match shape {shape} length {length}")
        tensor = self.alloc_tensor(shape)
        host = (ctypes.c_float * length)(*(float(value) for value in values))
        ffi._check(ffi._core.i_cuda_copy_from_host(tensor.data, host, length))
        return tensor

    def download_tensor(self, tensor_id: str) -> dict[str, Any]:
        tensor = self.get_tensor(tensor_id)
        host = (ctypes.c_float * tensor.length)()
        ffi._check(ffi._core.i_cuda_copy_to_host(host, tensor.data, tensor.length))
        return {"id": tensor.tensor_id, "shape": list(tensor.shape), "data": [float(host[i]) for i in range(tensor.length)]}

    def get_tensor(self, tensor_id: str) -> StoredTensor:
        with self.lock:
            tensor = self.tensors.get(tensor_id)
        if tensor is None:
            raise KeyError(f"unknown tensor id {tensor_id!r}")
        return tensor

    def delete_tensor(self, tensor_id: str) -> None:
        with self.lock:
            tensor = self.tensors.pop(tensor_id, None)
        if tensor is not None:
            ffi._core.i_cuda_free(tensor.data)

    def program(self, program_hash: str, source: str) -> Program:
        with self.lock:
            cached = self.programs.get(program_hash)
        if cached is not None:
            return cached

        program = compile_program(source)
        with self.lock:
            existing = self.programs.setdefault(program_hash, program)
        return existing

    def close(self) -> None:
        with self.lock:
            tensors = list(self.tensors.values())
            programs = list(self.programs.values())
            self.tensors.clear()
            self.programs.clear()
        for tensor in tensors:
            ffi._core.i_cuda_free(tensor.data)
        for program in programs:
            try:
                os.remove(program.path)
            except OSError:
                pass


def compile_program(source: str) -> Program:
    stem = f"ilang_remote_{uuid.uuid4().hex}"
    temp_dir = Path(tempfile.gettempdir())
    source_path = temp_dir / f"{stem}.cu"
    dylib_path = temp_dir / f"{stem}.{_dylib_ext()}"
    source_path.write_text(source)
    nvcc = os.environ.get("ILANG_NVCC", "nvcc")
    try:
        result = subprocess.run(
            [
                nvcc,
                "-O3",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                "--diag-suppress=177",
                "--cudart=shared",
                str(source_path),
                "-o",
                str(dylib_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            source_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed with status {result.returncode}: {result.stderr.strip()}")
    return Program(dylib_path)


def _dylib_ext() -> str:
    if os.name == "nt":
        return "dll"
    if os.uname().sysname == "Darwin":
        return "dylib"
    return "so"


class Handler(BaseHTTPRequestHandler):
    server: "RemoteServer"

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.quiet:
            return
        super().log_message(format, *args)

    def _handle(self, method: str) -> None:
        try:
            response = self._route(method)
            self._send(200, response)
        except KeyError as err:
            self._send(404, {"error": str(err)})
        except Exception as err:
            self._send(500, {"error": str(err)})

    def _route(self, method: str) -> Any:
        path = urlparse(self.path).path
        state = self.server.state

        if method == "GET" and path == "/health":
            return {"ok": True, "backend": "cuda", "api": 1}

        if method == "POST" and path == "/tensors/upload":
            payload = self._read_json()
            tensor = state.upload_tensor(tuple(int(dim) for dim in payload["shape"]), list(payload["data"]))
            return {"id": tensor.tensor_id, "shape": list(tensor.shape)}

        if path.startswith("/tensors/"):
            tensor_id = path.removeprefix("/tensors/")
            if method == "GET":
                return state.download_tensor(tensor_id)
            if method == "DELETE":
                state.delete_tensor(tensor_id)
                return {"ok": True}

        if method == "POST" and path == "/exec":
            payload = self._read_json()
            inputs = [state.get_tensor(str(tensor_id)) for tensor_id in payload["inputs"]]
            program = state.program(str(payload["program_hash"]), str(payload["cuda_source"]))
            outputs = [state.alloc_tensor(shape) for shape in program.output_shapes(inputs)]
            program.exec(inputs, outputs)
            return {"outputs": [{"id": tensor.tensor_id, "shape": list(tensor.shape)} for tensor in outputs]}

        raise KeyError(f"unknown endpoint {method} {path}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RemoteServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], quiet: bool = False) -> None:
        super().__init__(address, Handler)
        self.state = State()
        self.quiet = quiet

    def server_close(self) -> None:
        try:
            self.state.close()
        finally:
            super().server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an experimental remote 𝚒 CUDA server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7088)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    server = RemoteServer((args.host, args.port), quiet=args.quiet)
    print(f"serving remote i CUDA backend on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

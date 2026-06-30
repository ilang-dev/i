from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def is_remote_device(value: Any) -> bool:
    return bool(getattr(value, "_is_ilang_remote_device", False))


def is_remote_tensor(value: Any) -> bool:
    return is_remote_device(getattr(value, "device", None)) and hasattr(value, "_remote_id")


@dataclass(frozen=True)
class RemoteDevice:
    url: str
    timeout: float | None = 60.0

    _is_ilang_remote_device: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", self.url.rstrip("/"))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as err:
            message = err.reason
            try:
                decoded = json.loads(err.read().decode("utf-8"))
                message = decoded.get("error", message)
            except Exception:
                pass
            raise RuntimeError(f"remote i server error: {message}") from err
        except URLError as err:
            raise RuntimeError(f"remote i server unavailable at {self.url}: {err.reason}") from err

        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def health(self) -> Any:
        return self._request("GET", "/health")

    def upload(self, tensor: Any) -> Any:
        from ..tensor import Device, Tensor

        if is_remote_tensor(tensor):
            if tensor.device == self:
                return tensor
            tensor = tensor.to(Device.CPU)
        elif getattr(tensor, "device", Device.CPU) is not Device.CPU:
            tensor = tensor.to(Device.CPU)
        elif not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)

        response = self._request(
            "POST",
            "/tensors/upload",
            {"shape": list(tensor.shape), "data": tensor.data},
        )
        return Tensor._remote(tuple(response["shape"]), self, response["id"])

    def download(self, tensor: Any) -> Any:
        from ..tensor import Tensor

        if not is_remote_tensor(tensor):
            raise TypeError("download expects a remote i.Tensor")
        if tensor.device != self:
            raise TypeError("remote tensor belongs to a different remote device")

        response = self._request("GET", f"/tensors/{tensor._remote_id}")
        return Tensor(response["data"], shape=tuple(response["shape"]))

    def delete(self, tensor_id: str) -> None:
        try:
            self._request("DELETE", f"/tensors/{tensor_id}")
        except Exception:
            # Best-effort cleanup. The server also owns process-exit cleanup.
            pass

    def exec(self, component: Any, inputs: tuple[Any, ...], into: Any = None) -> Any:
        from ..tensor import Tensor

        if into is not None and not (
            into is Tensor or (isinstance(into, str) and into in {"tensor", "ilang", "i"})
        ):
            raise TypeError("remote execution returns remote i.Tensor outputs; download explicitly with .to(i.Device.CPU)")

        ids: list[str] = []
        for value in inputs:
            if not is_remote_tensor(value):
                raise TypeError("remote execution inputs must be remote i.Tensor values")
            if value.device != self:
                raise TypeError("all remote execution inputs must be on the same remote device")
            ids.append(value._remote_id)

        source = component._cuda_code()
        digest = sha256(source.encode("utf-8")).hexdigest()
        response = self._request(
            "POST",
            "/exec",
            {"program_hash": digest, "cuda_source": source, "inputs": ids},
        )
        outputs = [
            Tensor._remote(tuple(output["shape"]), self, output["id"])
            for output in response["outputs"]
        ]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __repr__(self) -> str:
        return f"Remote({self.url!r})"


class RemoteTensorOwner:
    def __init__(self, device: RemoteDevice, tensor_id: str) -> None:
        self.device: RemoteDevice | None = device
        self.tensor_id: str | None = tensor_id

    def __del__(self) -> None:
        device = getattr(self, "device", None)
        tensor_id = getattr(self, "tensor_id", None)
        if device is not None and tensor_id is not None:
            self.device = None
            self.tensor_id = None
            device.delete(tensor_id)


def exec_remote(component: Any, device: RemoteDevice, inputs: tuple[Any, ...], into: Any = None) -> Any:
    return device.exec(component, inputs, into=into)

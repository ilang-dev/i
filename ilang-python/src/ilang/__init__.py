"""Python front-end for 𝚒."""

from .component import Component
from .tensor import Device, Tensor

class _i:
    Component = Component
    Tensor = Tensor
    Device = Device

    @property
    def I(self) -> Component:
        return Component.I

    def __call__(self, expr: str) -> Component:
        return Component(expr)

i = _i()

__all__ = ["Component", "Device", "Tensor", "i"]

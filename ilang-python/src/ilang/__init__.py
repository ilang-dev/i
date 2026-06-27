"""Python front-end for 𝚒."""

from .component import Bench, Component, I
from .tensor import Device, Tensor

class _i:
    Component = Component
    Tensor = Tensor
    Device = Device

    @property
    def I(self) -> Component:
        return I

    def __call__(self, expr: str) -> Component:
        return Component(expr)

i = _i()

__all__ = ["Bench", "Component", "Device", "Tensor", "I", "i"]

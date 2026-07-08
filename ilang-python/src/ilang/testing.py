from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from math import floor, log10, sqrt
from time import perf_counter
from typing import TypeVar
import string

from . import i
from .tensor import Tensor

__all__ = ["Bench", "bench"]

T = TypeVar("T")


@dataclass
class Bench:
    mean: timedelta
    std: timedelta
    n_warmups: int
    n_runs: int
    runs: list[timedelta]

    def _human_time(self) -> str:
        if self.mean.total_seconds() == 0:
            mean_order = -6
        else:
            mean_order = floor(log10(self.mean.total_seconds()))
        if mean_order <= -6:
            scale = 9
            unit = "ns"
        elif mean_order <= -3:
            scale = 6
            unit = "μs"
        elif mean_order <= 0:
            scale = 3
            unit = "ms"
        else:
            scale = 0
            unit = "s"
        mean_str = f"{round(self.mean.total_seconds() * 10**scale)}"
        std_str = f"{round(self.std.total_seconds() * 10**scale)}"
        return f"{mean_str}±{std_str} {unit}"

    def __repr__(self) -> str:
        return f"{self._human_time()}, warmups = {self.n_warmups}, runs = {self.n_runs}"


def bench(fn: Callable[[], T], n_warmups: int = 10, n_runs: int = 100) -> Bench:
    for _ in range(n_warmups):
        fn()

    runs: list[timedelta] = []
    for _ in range(n_runs):
        start = perf_counter()
        fn()
        end = perf_counter()
        runs.append(timedelta(seconds=end - start))

    mean = timedelta(seconds=sum(run.total_seconds() for run in runs) / len(runs))
    if len(runs) < 2:
        std = timedelta(0)
    else:
        std = timedelta(
            seconds=sqrt(
                1
                / (len(runs) - 1)
                * sum((run - mean).total_seconds() ** 2 for run in runs)
            )
        )

    return Bench(
        mean=mean,
        std=std,
        n_warmups=n_warmups,
        n_runs=n_runs,
        runs=runs,
    )

def allclose(val: Tensor, ref: Tensor, rtol=1e-05, atol=1e-08):
    """elementwise_all(absolute(val - ref) <= (atol + rtol * absolute(ref)))"""

    assert val.shape == ref.shape, f"{val.shape=} != {ref.shape=}"
    assert len(val.shape) <= len(string.ascii_lowercase), "really?"
    n = len(val.shape)
    index = "." if n == 0 else string.ascii_lowercase[:n]

    sub_ = i(f"{index}-{index}~{index}")
    max_ = i(f"{index}>{index}~{index}")
    neg_ = i(f"-{index}~{index}")
    abs_ = (i.I & neg_) >> max_
    mul_ = i(f".*{index}~{index}")
    add_ = i(f".+{index}~{index}")
    minr_ = i(f"<{index}~.")

    left = sub_(val, ref)
    left = abs_(left)

    right = ref
    right = abs_(ref)
    right = mul_(rtol, right)
    right = add_(atol, right)

    diff = sub_(right, left)
    m = minr_(diff)()

    return m.data[0] >= 0


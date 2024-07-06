from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from check_shapes import check_shapes

from jopfra.paths import MiscDir
from jopfra.types import AnyNDArray


@dataclass(order=True, frozen=True)
class Evaluation:
    problem: Problem
    x: AnyNDArray
    loss: AnyNDArray
    grads: AnyNDArray

    @check_shapes(
        "self.x: [batch_shape..., n_inputs]",
        "self.loss: [batch_shape...]",
        "self.grads: [batch_shape..., n_inputs]",
    )
    def __post_init__(self) -> None:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.loss.shape

    @property
    def n_inputs(self) -> int:
        return self.problem.n_inputs


class Problem(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def n_inputs(self) -> int: ...

    @property
    def domain_lower(self) -> AnyNDArray: ...

    @property
    def domain_upper(self) -> AnyNDArray: ...

    @property
    def known_optima(self) -> Collection[AnyNDArray]: ...

    @check_shapes(
        "x: [batch..., n_inputs]",
        "return: [batch...]",
    )
    def __call__(self, x: AnyNDArray) -> Evaluation: ...

    @check_shapes(
        "x: [n_inputs]",
    )
    def plot(self, dest: MiscDir, x: AnyNDArray) -> None: ...

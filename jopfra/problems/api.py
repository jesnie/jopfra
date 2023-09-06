from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Protocol

import numpy as np
import torch as tc
from check_shapes import check_shapes, get_check_shapes

from jopfra.types import AnyNDArray

check_problem_shapes = check_shapes(
    "x: [batch..., n_inputs]",
    "return[0]: [batch...]  # Loss",
    "return[1]: [batch..., n_inputs]  # Gradients",
)


class ProblemFunc(Protocol):
    def __call__(self, x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
        ...

    @property
    def __name__(self) -> str:
        ...


@dataclass(order=True, frozen=True)
class Problem:
    name: str
    domain_lower: AnyNDArray
    domain_upper: AnyNDArray
    known_optima: Collection[AnyNDArray]
    func: ProblemFunc

    @check_shapes(
        "self.domain_lower: [n_inputs]",
        "self.domain_upper: [n_inputs]",
        "self.known_optima: [n_optima, n_inputs]",
    )
    def __post_init__(self) -> None:
        assert check_problem_shapes is get_check_shapes(self.func)

    @check_problem_shapes
    def __call__(self, x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
        return self.func(x)

    @property
    def n_inputs(self) -> int:
        (n_inputs,) = self.domain_lower.shape
        return n_inputs


problems: dict[str, Problem] = {}


def problem(
    domain_lower: AnyNDArray | Sequence[float],
    domain_upper: AnyNDArray | Sequence[float],
    known_optima: Collection[AnyNDArray | Sequence[float]],
    *,
    name: str | None = None,
) -> Callable[[ProblemFunc], Problem]:
    def _wrap(func: ProblemFunc) -> Problem:
        nonlocal name
        name = name or func.__name__
        p = Problem(
            name,
            np.asarray(domain_lower),
            np.asarray(domain_upper),
            [np.asarray(o) for o in known_optima],
            check_problem_shapes(func),
        )
        problems[name] = p
        return p

    return _wrap


class TorchProblemFunc(Protocol):
    def __call__(self, x: tc.Tensor) -> tc.Tensor:
        ...

    @property
    def __name__(self) -> str:
        ...


def torch_problem(
    domain_lower: AnyNDArray | Sequence[float],
    domain_upper: AnyNDArray | Sequence[float],
    known_optima: Collection[AnyNDArray | Sequence[float]],
    *,
    name: str | None = None,
) -> Callable[[TorchProblemFunc], Problem]:
    def _wrap(func: TorchProblemFunc) -> Problem:
        @wraps(func)
        def __wrap(x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
            tx = tc.tensor(x, requires_grad=True)
            rx = func(tx)
            rx.backward(tc.zeros(tx.shape[:-1], dtype=tx.dtype))  # type: ignore[no-untyped-call]
            return (
                rx.detach().numpy(),
                tx.grad.detach().numpy(),  # type: ignore[union-attr]
            )

        return problem(
            domain_lower=domain_lower,
            domain_upper=domain_upper,
            known_optima=known_optima,
            name=name,
        )(__wrap)

    return _wrap

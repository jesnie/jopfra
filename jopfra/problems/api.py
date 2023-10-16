from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Protocol

import numpy as np
import torch as tc
from check_shapes import check_shapes, get_check_shapes

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
    def name(self) -> str:
        ...

    @property
    def n_inputs(self) -> int:
        ...

    @property
    def domain_lower(self) -> AnyNDArray:
        ...

    @property
    def domain_upper(self) -> AnyNDArray:
        ...

    @property
    def known_optima(self) -> Collection[AnyNDArray]:
        ...

    @check_shapes(
        "x: [batch..., n_inputs]",
        "return: [batch...]",
    )
    def __call__(self, x: AnyNDArray) -> Evaluation:
        ...

    @check_shapes(
        "x: [n_inputs]",
    )
    def plot(self, dest: MiscDir, x: AnyNDArray) -> None:
        ...


problems: dict[str, Problem] = {}


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


check_plot_shapes = check_shapes(
    "x: [n_inputs]",
)


class PlotFunc(Protocol):
    def __call__(self, dest: MiscDir, x: AnyNDArray) -> None:
        ...


@dataclass(order=True, frozen=True)
class FuncProblem:
    name: str
    domain_lower: AnyNDArray
    domain_upper: AnyNDArray
    known_optima: Collection[AnyNDArray]
    func: ProblemFunc
    plot_: PlotFunc

    @check_shapes(
        "self.domain_lower: [n_inputs]",
        "self.domain_upper: [n_inputs]",
        "self.known_optima: [n_optima, n_inputs]",
    )
    def __post_init__(self) -> None:
        assert check_problem_shapes is get_check_shapes(self.func), get_check_shapes(self.func)
        assert check_plot_shapes is get_check_shapes(self.plot_), get_check_shapes(self.plot_)

    @get_check_shapes(Problem.__call__)
    def __call__(self, x: AnyNDArray) -> Evaluation:
        loss, grads = self.func(x)
        return Evaluation(self, x, loss, grads)

    @get_check_shapes(Problem.plot)
    def plot(self, dest: MiscDir, x: AnyNDArray) -> None:
        self.plot_(dest, x)

    @property
    def n_inputs(self) -> int:
        (n_inputs,) = self.domain_lower.shape
        return n_inputs


def problem(
    domain_lower: AnyNDArray | Sequence[float],
    domain_upper: AnyNDArray | Sequence[float],
    known_optima: Collection[AnyNDArray | Sequence[float]],
    *,
    name: str | None = None,
    plot: PlotFunc = check_plot_shapes(lambda dest, x: None),
) -> Callable[[ProblemFunc], Problem]:
    def _wrap(func: ProblemFunc) -> Problem:
        nonlocal name
        name = name or func.__name__
        p = FuncProblem(
            name,
            np.asarray(domain_lower),
            np.asarray(domain_upper),
            [np.asarray(o) for o in known_optima],
            check_problem_shapes(func),
            plot,
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


class TorchPlotFunc(Protocol):
    def __call__(self, dest: MiscDir, x: tc.Tensor) -> None:
        ...


def torch_problem(
    domain_lower: AnyNDArray | Sequence[float],
    domain_upper: AnyNDArray | Sequence[float],
    known_optima: Collection[AnyNDArray | Sequence[float]],
    *,
    name: str | None = None,
    plot: TorchPlotFunc = check_plot_shapes(lambda dest, x: None),
) -> Callable[[TorchProblemFunc], Problem]:
    def _wrap(func: TorchProblemFunc) -> Problem:
        @wraps(func)
        def _call(x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
            tx = tc.tensor(x, requires_grad=True)
            rx = func(tx)
            rx.backward(tc.ones(tx.shape[:-1], dtype=tx.dtype))  # type: ignore[no-untyped-call]
            return (
                rx.detach().numpy(),
                tx.grad.detach().numpy(),  # type: ignore[union-attr]
            )

        @wraps(plot)
        def _plot(dest: MiscDir, x: AnyNDArray) -> None:
            plot(dest, tc.tensor(x))

        return problem(
            domain_lower=domain_lower,
            domain_upper=domain_upper,
            known_optima=known_optima,
            name=name,
            plot=_plot,
        )(_call)

    return _wrap

from collections.abc import Iterator
from typing import Callable, Iterable, TypeAlias

import numpy as np
import torch as tc

from jopfra.minimisers.api import IterMinimiser, SingleMinimiser, iter_minimisers
from jopfra.minimisers.sobol import sobol
from jopfra.problems.api import Evaluation, Problem
from jopfra.problems.utils import wrap_domain

TorchOptimiserFactory: TypeAlias = Callable[[Iterable[tc.Tensor]], tc.optim.Optimizer]


class TorchOptimiser(IterMinimiser):
    def __init__(self, factory: TorchOptimiserFactory, initial_value: SingleMinimiser) -> None:
        self._factory = factory
        self._initial_value = initial_value

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        y = self._initial_value.single_minimise(problem, batch_shape)
        x = tc.tensor(y.x, requires_grad=True)
        optimizer = self._factory([x])
        while True:
            yield y
            optimizer.zero_grad()
            x.grad = tc.tensor(y.grads, dtype=x.dtype)
            optimizer.step()
            np_x = x.detach().numpy()
            np_x = wrap_domain(problem, np_x)
            with tc.no_grad():
                x[...] = tc.tensor(np_x, dtype=x.dtype)  # type: ignore[index]
            y = problem(np_x)


iter_minimisers["tc_adadelta"] = TorchOptimiser(tc.optim.Adadelta, sobol)
iter_minimisers["tc_adagrad"] = TorchOptimiser(tc.optim.Adagrad, sobol)
iter_minimisers["tc_adam"] = TorchOptimiser(tc.optim.Adam, sobol)
iter_minimisers["tc_rmsprop"] = TorchOptimiser(tc.optim.RMSprop, sobol)
iter_minimisers["tc_rprop"] = TorchOptimiser(tc.optim.Rprop, sobol)
iter_minimisers["tc_sgd_1"] = TorchOptimiser(lambda xs: tc.optim.SGD(xs, lr=1e-1), sobol)
iter_minimisers["tc_sgd_2"] = TorchOptimiser(lambda xs: tc.optim.SGD(xs, lr=1e-2), sobol)
iter_minimisers["tc_sgd_3"] = TorchOptimiser(lambda xs: tc.optim.SGD(xs, lr=1e-3), sobol)

CallbackTorchOptimiserFactory: TypeAlias = Callable[[Iterable[tc.Tensor]], tc.optim.Optimizer]


class CallbackTorchOptimiser(IterMinimiser):
    def __init__(
        self, factory: CallbackTorchOptimiserFactory, initial_value: SingleMinimiser
    ) -> None:
        self._factory = factory
        self._initial_value = initial_value

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        y = self._initial_value.single_minimise(problem, batch_shape)
        x = tc.tensor(y.x, requires_grad=True)
        optimizer = self._factory([x])
        while True:
            yield y

            def loss() -> float:
                nonlocal x, y
                optimizer.zero_grad()
                np_x = x.detach().numpy()
                np_x = wrap_domain(problem, np_x)
                y = problem(np_x)
                with tc.no_grad():
                    x[...] = tc.tensor(np_x, dtype=x.dtype)  # type: ignore[index]
                    x.grad = tc.tensor(y.grads, dtype=x.dtype)
                return float(np.sum(y.loss))

            optimizer.step(loss)


iter_minimisers["tc_lbfgs"] = CallbackTorchOptimiser(tc.optim.LBFGS, sobol)

import time
from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np
from check_shapes import check_shapes

from jopfra.paths import MiscDir
from jopfra.problems.api import Evaluation, Problem
from jopfra.types import AnyNDArray


@dataclass
class LoggingProblem:
    root: Problem
    start_ns: int = field(default_factory=time.perf_counter_ns)
    log: list[tuple[int, Evaluation]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def n_inputs(self) -> int:
        return self.root.n_inputs

    @property
    def domain_lower(self) -> AnyNDArray:
        return self.root.domain_lower

    @property
    def domain_upper(self) -> AnyNDArray:
        return self.root.domain_upper

    @property
    def known_optima(self) -> Collection[AnyNDArray]:
        return self.root.known_optima

    @check_shapes(
        "x: [batch..., n_inputs]",
        "return: [batch...]",
    )
    def __call__(self, x: AnyNDArray) -> Evaluation:
        y = self.root(x)
        t = time.perf_counter_ns()
        self.log.append((t - self.start_ns, y))
        return y

    @check_shapes(
        "x: [n_inputs]",
    )
    def plot(self, dest: MiscDir, x: AnyNDArray) -> None:
        self.root.plot(dest, x)

    @check_shapes(
        "return: [n_calls]",
    )
    def min_losses(self) -> AnyNDArray:
        min_losses = [np.min(y.loss) for _, y in self.log]
        return np.array(min_losses)

    @check_shapes(
        "return: [n_calls]",
    )
    def n_calls(self) -> AnyNDArray:
        return np.arange(len(self.log)) + 1

    @check_shapes(
        "return: [n_calls]",
    )
    def n_evals(self) -> AnyNDArray:
        batch_size = [np.prod(y.shape, dtype=np.int32) for _, y in self.log]
        return np.cumsum(batch_size)

    @check_shapes(
        "return: [n_calls]",
    )
    def times(self) -> AnyNDArray:
        return np.array([t for t, _ in self.log])

from collections.abc import Iterator
from typing import Final

import numpy as np

from jopfra.api import Evaluation, Problem
from jopfra.minimisers.api import IterMinimiser, SingleMinimiser


class Random(IterMinimiser, SingleMinimiser):
    def __init__(self, seed: int | None = 20230907) -> None:
        self._seed = seed

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        rng = np.random.default_rng(self._seed)
        while True:
            x = rng.random(batch_shape + (problem.n_inputs,))
            x *= problem.domain_upper - problem.domain_lower
            x += problem.domain_lower
            x = np.reshape(x, batch_shape + (problem.n_inputs,))
            y = problem(x)
            yield y

    def single_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Evaluation:
        return next(self.iter_minimise(problem, batch_shape))


random: Final[Random] = Random()

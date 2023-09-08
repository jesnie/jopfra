from collections.abc import Iterator

import numpy as np

from jopfra.minimisers.api import Minimiser
from jopfra.problems.api import Evaluation, Problem


class Random(Minimiser):
    def __init__(self, seed: int | None = 20230907) -> None:
        self._seed = seed

    def minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        rng = np.random.default_rng(self._seed)
        while True:
            x = rng.random(batch_shape + (problem.n_inputs,))
            x *= problem.domain_upper - problem.domain_lower
            x += problem.domain_lower
            x = np.reshape(x, batch_shape + (problem.n_inputs,))
            y = problem(x)
            yield y


random = Random()

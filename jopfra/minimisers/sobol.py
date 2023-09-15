from collections.abc import Iterator
from typing import Final

import torch as tc

from jopfra.flatten import Flattener
from jopfra.minimisers.api import Minimiser
from jopfra.problems.api import Evaluation, Problem
from jopfra.types import AnyNDArray


class Sobol(Minimiser):
    def minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        flat = Flattener(batch_shape)
        soboleng = tc.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
            dimension=problem.n_inputs
        )
        while True:
            x: AnyNDArray = soboleng.draw(flat.size).numpy()
            x *= problem.domain_upper - problem.domain_lower
            x += problem.domain_lower
            x = flat.unflatten(x)
            y = problem(x)
            yield y


sobol: Final[Sobol] = Sobol()

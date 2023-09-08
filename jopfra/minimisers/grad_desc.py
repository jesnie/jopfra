from collections.abc import Iterator

from jopfra.minimisers.api import Minimiser, minimisers
from jopfra.minimisers.min import Min, sobol_min
from jopfra.minimisers.sobol import sobol
from jopfra.problems.api import Evaluation, Problem


class GradDesc(Minimiser):
    def __init__(self, initial_value: Minimiser, learning_rate: float) -> None:
        assert learning_rate > 0.0, learning_rate
        self._initial_value = initial_value
        self._learning_rate = learning_rate

    def minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        for y in self._initial_value.minimise(problem, batch_shape):
            while True:
                yield y
                x = y.x - self._learning_rate * y.grads
                y = problem(x)


minimisers["grad_desc_1"] = GradDesc(sobol, 1e-1)
minimisers["grad_desc_2"] = GradDesc(sobol, 1e-2)
minimisers["grad_desc_3"] = GradDesc(sobol, 1e-3)
minimisers["sobol_min_grad_desc_1"] = GradDesc(sobol_min, 1e-1)
minimisers["sobol_min_grad_desc_2"] = GradDesc(sobol_min, 1e-2)
minimisers["sobol_min_grad_desc_3"] = GradDesc(sobol_min, 1e-3)
minimisers["sobol_min_grad_desc_1_min"] = Min(GradDesc(sobol_min, 1e-1), scale=10, memory=True)
minimisers["sobol_min_grad_desc_2_min"] = Min(GradDesc(sobol_min, 1e-2), scale=10, memory=True)
minimisers["sobol_min_grad_desc_3_min"] = Min(GradDesc(sobol_min, 1e-3), scale=10, memory=True)

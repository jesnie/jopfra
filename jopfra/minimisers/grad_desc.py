from collections.abc import Iterator

from jopfra.api import Evaluation, Problem
from jopfra.minimisers.api import IterMinimiser, SingleMinimiser, iter_minimisers
from jopfra.minimisers.min import IterMin, sobol_min_init
from jopfra.minimisers.sobol import sobol
from jopfra.problems.utils import wrap_domain


class GradDesc(IterMinimiser):
    def __init__(self, initial_value: SingleMinimiser, learning_rate: float) -> None:
        assert learning_rate > 0.0, learning_rate
        self._initial_value = initial_value
        self._learning_rate = learning_rate

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        y = self._initial_value.single_minimise(problem, batch_shape)
        while True:
            yield y
            x = y.x - self._learning_rate * y.grads
            x = wrap_domain(problem, x)
            y = problem(x)


iter_minimisers["grad_desc_1"] = GradDesc(sobol, 1e-1)
iter_minimisers["grad_desc_2"] = GradDesc(sobol, 1e-2)
iter_minimisers["grad_desc_3"] = GradDesc(sobol, 1e-3)
iter_minimisers["sobol_min_grad_desc_1"] = GradDesc(sobol_min_init, 1e-1)
iter_minimisers["sobol_min_grad_desc_2"] = GradDesc(sobol_min_init, 1e-2)
iter_minimisers["sobol_min_grad_desc_3"] = GradDesc(sobol_min_init, 1e-3)
iter_minimisers["sobol_min_grad_desc_1_min"] = IterMin(
    GradDesc(sobol_min_init, 1e-1), scale=10, memory=True
)
iter_minimisers["sobol_min_grad_desc_2_min"] = IterMin(
    GradDesc(sobol_min_init, 1e-2), scale=10, memory=True
)
iter_minimisers["sobol_min_grad_desc_3_min"] = IterMin(
    GradDesc(sobol_min_init, 1e-3), scale=10, memory=True
)

from collections.abc import Sequence

import torch as tc
from check_shapes import check_shapes

from jopfra.problems.api import Problem, torch_problem


@check_shapes(
    "domain_lower: [n_inputs]",
    "domain_upper: [n_inputs]",
    "minimum: [n_inputs]",
)
def make_quadratic(
    name: str,
    domain_lower: Sequence[float],
    domain_upper: Sequence[float],
    minimum: Sequence[float],
) -> Problem:
    tc_minimum = tc.tensor(minimum)

    @torch_problem(
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        known_optima=[minimum],
        name=name,
    )
    def quadratic(x: tc.Tensor) -> tc.Tensor:
        return tc.sum((x - tc_minimum) ** 2, dim=-1)

    return quadratic


make_quadratic("quadratic_base", [-1.0], [1.0], [0.0])
make_quadratic("quadratic_off_center", [-1.0], [1.0], [0.5])
for d in [2, 3, 5]:
    make_quadratic(f"quadratic_{d}d", d * [-1.0], d * [1.0], d * [0.0])

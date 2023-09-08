import numpy as np
import pytest

from jopfra.minimisers.api import minimisers
from jopfra.problems.quadratic import make_quadratic


@pytest.mark.parametrize("minimiser_name", minimisers)
def test_minimiser__batch_shape(minimiser_name: str) -> None:
    problem = make_quadratic("test_problem", [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    minimiser = minimisers[minimiser_name]

    y = next(minimiser.minimise(problem, (4, 2)))

    assert (4, 2) == y.shape
    assert 3 == y.n_inputs
    assert (4, 2, 3) == y.x.shape


@pytest.mark.parametrize("minimiser_name", minimisers)
def test_minimiser__find_minimum(minimiser_name: str) -> None:
    minimum = [0.2, 0.8, 0.5]
    problem = make_quadratic("test_problem", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], minimum)
    minimiser = minimisers[minimiser_name]

    for i, y in enumerate(minimiser.minimise(problem, ())):
        if i >= 10_000:
            raise AssertionError
        max_dist = np.max(np.abs(y.x - minimum))
        if max_dist < 0.05:
            break

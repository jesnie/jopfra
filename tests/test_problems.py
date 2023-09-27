import numpy as np
import pytest

from jopfra.problems.api import Problem, problems
from jopfra.problems.utils import get_domain_corners, get_sobol_samples


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__domain(problem: Problem) -> None:
    if problem.n_inputs <= 10:
        x = get_domain_corners(problem)
    else:
        x = get_sobol_samples(problem, 1_000)
    y = problem(x)
    assert y.problem is problem
    assert y.x is x
    assert np.all(np.isfinite(y.loss))
    assert np.all(np.isfinite(y.grads))


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__batching(problem: Problem) -> None:
    h = 4
    w = 3
    x = np.reshape(get_sobol_samples(problem, h * w), [h, w, -1])
    y = problem(x)
    for i in range(h):
        for j in range(w):
            yij = problem(x[i, j])
            np.testing.assert_equal(y.loss[i, j], yij.loss)
            np.testing.assert_equal(y.grads[i, j], yij.grads)


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__optima(problem: Problem) -> None:
    if not problem.known_optima:
        return

    baseline_x = get_sobol_samples(problem, 1_000)
    if problem.n_inputs <= 10:
        baseline_x = np.concatenate(
            [
                get_domain_corners(problem),
                baseline_x,
            ],
            axis=0,
        )
    baseline_y = problem(baseline_x)
    min_baseline_loss = np.min(baseline_y.loss)

    optimal_x = np.array(problem.known_optima)
    optimal_y = problem(optimal_x)
    min_optimal_loss = np.min(optimal_y.loss)

    assert min_baseline_loss >= min_optimal_loss
    np.testing.assert_allclose(optimal_y.loss, min_optimal_loss)

import numpy as np
import pytest
import torch as tc
from check_shapes import check_shapes

from jopfra.problems.api import Problem, problems
from jopfra.types import AnyNDArray


@check_shapes(
    "problem.domain_lower: [n_inputs]",
    "problem.domain_upper: [n_inputs]",
    "return: [n_corners, n_inputs]",
)
def get_domain_corners(problem: Problem) -> AnyNDArray:
    n_inputs = problem.n_inputs
    domain = np.stack([problem.domain_lower, problem.domain_upper])
    result = np.zeros(n_inputs * [1] + [n_inputs], dtype=domain.dtype)
    for i in range(n_inputs):
        dim = np.zeros([2, n_inputs], dtype=domain.dtype)
        dim[:, i] = domain[:, i]
        dim = np.reshape(dim, i * [1] + [2] + (n_inputs - 1 - i) * [1] + [n_inputs])
        result = result + dim
    return np.reshape(result, [-1, n_inputs])


@check_shapes(
    "problem.domain_lower: [n_inputs]",
    "problem.domain_upper: [n_inputs]",
    "return: [n_samples, n_inputs]",
)
def get_sobol_samples(problem: Problem, n_samples: int) -> AnyNDArray:
    soboleng = tc.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=problem.n_inputs
    )
    samples: AnyNDArray = soboleng.draw(n_samples).numpy()
    samples *= problem.domain_upper - problem.domain_lower
    samples += problem.domain_lower
    return samples


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__domain(problem: Problem) -> None:
    x = get_domain_corners(problem)
    loss, grads = problem(x)
    assert np.all(np.isfinite(loss))
    assert np.all(np.isfinite(grads))


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__batching(problem: Problem) -> None:
    h = 4
    w = 3
    x = np.reshape(get_sobol_samples(problem, h * w), [h, w, -1])
    loss, grads = problem(x)
    for i in range(h):
        for j in range(w):
            loss_ij, grads_ij = problem(x[i, j])
            np.testing.assert_equal(loss[i, j], loss_ij)
            np.testing.assert_equal(grads[i, j], grads_ij)


@pytest.mark.parametrize("problem", problems.values(), ids=lambda p: p.name)
def test_problem__optima(problem: Problem) -> None:
    baseline_x = np.concatenate(
        [
            get_domain_corners(problem),
            get_sobol_samples(problem, 10),
        ],
        axis=0,
    )
    baseline_loss, _ = problem(baseline_x)
    min_baseline_loss = np.min(baseline_loss)

    optimal_x = np.array(problem.known_optima)
    optimal_loss, _ = problem(optimal_x)
    min_optimal_loss = np.min(optimal_loss)

    assert min_baseline_loss >= min_optimal_loss
    np.testing.assert_allclose(optimal_loss, min_optimal_loss)

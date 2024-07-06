from unittest.mock import MagicMock

import numpy as np

from jopfra.api import Problem
from jopfra.problems.utils import (
    clip_domain,
    get_domain_corners,
    get_sobol_samples,
    pretty_exp,
    wrap_domain,
)


def test_get_domain_corners() -> None:
    problem = MagicMock(Problem)
    problem.n_inputs = 3
    problem.domain_lower = np.array([-2.0, -1.0, 1.0])
    problem.domain_upper = np.array([-1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        [
            [-2.0, -1.0, 1.0],
            [-2.0, -1.0, 2.0],
            [-2.0, 1.0, 1.0],
            [-2.0, 1.0, 2.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 2.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, 2.0],
        ],
        get_domain_corners(problem),
    )


def test_get_sobol_samples() -> None:
    problem = MagicMock(Problem)
    problem.n_inputs = 3
    problem.domain_lower = np.array([-2.0, -1.0, 1.0])
    problem.domain_upper = np.array([-1.0, 1.0, 2.0])
    x = get_sobol_samples(problem, 10)
    assert (10, 3) == x.shape
    assert np.all((problem.domain_lower <= x) & (x <= problem.domain_upper))


def test_clip_domain() -> None:
    problem = MagicMock(Problem)
    problem.domain_lower = np.array([-2.0, 1.0])
    problem.domain_upper = np.array([-1.0, 2.0])
    np.testing.assert_allclose(
        [
            [-2.0, 1.0],
            [-2.0, 1.0],
            [-1.5, 1.5],
            [-1.0, 2.0],
            [-1.0, 2.0],
        ],
        clip_domain(
            problem,
            np.array(
                [
                    [-3.1, -0.1],
                    [-2.1, 0.9],
                    [-1.5, 1.5],
                    [-0.9, 2.1],
                    [0.1, 3.1],
                ]
            ),
        ),
    )


def test_wrap_domain() -> None:
    problem = MagicMock(Problem)
    problem.domain_lower = np.array([-2.0, 1.0])
    problem.domain_upper = np.array([-1.0, 2.0])
    np.testing.assert_allclose(
        [
            [-1.1, 1.9],
            [-1.9, 1.1],
            [-1.5, 1.5],
            [-1.1, 1.9],
            [-1.9, 1.1],
        ],
        wrap_domain(
            problem,
            np.array(
                [
                    [-3.1, -0.1],
                    [-2.1, 0.9],
                    [-1.5, 1.5],
                    [-0.9, 2.1],
                    [0.1, 3.1],
                ]
            ),
        ),
    )


def test_pretty_exp() -> None:
    assert [] == list(pretty_exp(0))
    assert [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] == list(pretty_exp(10))

    l = []
    for _, i in zip(range(5), pretty_exp()):
        l.append(i)
    assert [1, 2, 5, 10, 20] == l

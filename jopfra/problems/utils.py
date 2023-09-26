from collections.abc import Iterator

import numpy as np
import torch as tc
from check_shapes import check_shapes

from jopfra.problems.api import Problem
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


@check_shapes(
    "problem.domain_lower: [n_inputs]",
    "problem.domain_upper: [n_inputs]",
    "x: [batch..., n_inputs]",
    "return: [batch..., n_inputs]",
)
def clip_domain(problem: Problem, x: AnyNDArray) -> AnyNDArray:
    return np.clip(x, problem.domain_lower, problem.domain_upper)


@check_shapes(
    "problem.domain_lower: [n_inputs]",
    "problem.domain_upper: [n_inputs]",
    "x: [batch..., n_inputs]",
    "return: [batch..., n_inputs]",
)
def wrap_domain(problem: Problem, x: AnyNDArray) -> AnyNDArray:
    x -= problem.domain_lower
    r = problem.domain_upper - problem.domain_lower
    x %= 2 * r
    x = np.where(x > r, 2 * r - x, x)
    x += problem.domain_lower
    return x


def pretty_exp(n: int | None = None) -> Iterator[int]:
    m = 0
    base = 1
    scales = (1, 2, 5)
    while True:
        for s in scales:
            if n is not None and m >= n:
                return
            yield s * base
            m += 1
        base *= 10

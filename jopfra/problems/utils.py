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

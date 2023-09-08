# http://www.sfu.ca/~ssurjano/index.html
from math import pi

import torch as tc

from jopfra.problems.api import Problem, torch_problem


def make_ackley(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-32.768 for _ in range(d)],
        domain_upper=[32.768 for _ in range(d)],
        known_optima=[[0.0 for _ in range(d)]],
        name=f"ackley_{d}d",
    )
    def ackley(x: tc.Tensor) -> tc.Tensor:
        c = tc.tensor(2 * pi)
        b = tc.tensor(0.2)
        a = tc.tensor(20)

        sum1 = tc.zeros_like(x[..., 0])
        sum2 = tc.zeros_like(x[..., 0])
        for i in range(d):
            xi = x[..., i]
            sum1 += xi**2
            sum2 += tc.cos(c * xi)

        term1 = -a * tc.exp(-b * tc.sqrt(sum1 / d))
        term2 = -tc.exp(sum2 / d)
        y = term1 + term2 + a + tc.e

        return y

    return ackley


for _d in [1, 2, 3]:
    make_ackley(_d)

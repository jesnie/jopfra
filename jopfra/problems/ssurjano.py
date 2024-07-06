# http://www.sfu.ca/~ssurjano/index.html
from collections.abc import Collection, Sequence
from math import pi

import numpy as np
import torch as tc
from check_shapes import check_shapes

from jopfra.api import Problem
from jopfra.problems.api import torch_problem
from jopfra.problems.utils import pretty_exp
from jopfra.types import AnyNDArray


def make_ackley(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-32.768 for _ in range(d)],
        domain_upper=[32.768 for _ in range(d)],
        known_optima=[[0.0 for _ in range(d)]],
        name=f"ackley_{d}d",
    )
    def ackley(x: tc.Tensor) -> tc.Tensor:
        c = tc.tensor(2 * tc.pi)
        b = tc.tensor(0.2)
        a = tc.tensor(20)
        sum1 = tc.sum(x**2, dim=-1)
        sum2 = tc.sum(tc.cos(c * x), dim=-1)
        term1 = -a * tc.exp(-b * tc.sqrt(sum1 / d))
        term2 = -tc.exp(sum2 / d)
        return term1 + term2 + a + tc.e

    return ackley


for _d in pretty_exp(10):
    make_ackley(_d)


@torch_problem(
    domain_lower=[-15.0, -3.0],
    domain_upper=[-5.0, 3.0],
    known_optima=[[-10.0, 1.0]],
)
def bukin_no6(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = 100 * tc.sqrt(tc.abs(x2 - 0.01 * (x1**2)))
    term2 = 0.01 * tc.abs(x1 + 10)
    return term1 + term2


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[
        [-1.3491, -1.3491],
        [-1.3491, 1.3491],
        [1.3491, -1.3491],
        [1.3491, 1.3491],
    ],
)
def cross_in_tray(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    fact1 = tc.sin(x1) * tc.sin(x2)
    fact2 = tc.exp(tc.abs(100 - tc.sqrt((x1**2) + (x2**2)) / tc.pi))
    return -0.0001 * tc.pow((tc.abs(fact1 * fact2) + 1), 0.1)


@torch_problem(
    domain_lower=[-5.12, -5.12],
    domain_upper=[5.12, 5.12],
    known_optima=[[0.0, 0.0]],
)
def drop_wave(x: tc.Tensor) -> tc.Tensor:
    x2 = tc.pow(x, 2.0)
    r2 = tc.sum(x2, dim=-1)
    frac1 = 1 + tc.cos(12 * tc.sqrt(r2))
    frac2 = 0.5 * r2 + 2
    return -frac1 / frac2


@torch_problem(
    domain_lower=[-512.0, -512.0],
    domain_upper=[512.0, 512.0],
    known_optima=[[512.0, 404.2319]],
)
def eggholder(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = -(x2 + 47) * tc.sin(tc.sqrt(tc.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * tc.sin(tc.sqrt(tc.abs(x1 - (x2 + 47))))
    return term1 + term2


@torch_problem(
    domain_lower=[0.5],
    domain_upper=[2.5],
    known_optima=[[0.5485634445513679]],
)
def gramacy_lee(x: tc.Tensor) -> tc.Tensor:
    x = x[..., 0]
    term1 = tc.sin(10 * tc.pi * x) / (2 * x)
    term2 = tc.pow(x - 1, 4.0)
    return term1 + term2


def make_griewank(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-600.0 for _ in range(d)],
        domain_upper=[600.0 for _ in range(d)],
        known_optima=[[0.0 for _ in range(d)]],
        name=f"griewank_{d}d",
    )
    def griewank(x: tc.Tensor) -> tc.Tensor:
        term1 = tc.sum(tc.pow(x, 2.0), dim=-1) / 4000.0
        term2 = tc.prod(tc.cos(x / tc.sqrt(tc.arange(d, dtype=x.dtype) + 1)), dim=-1)
        return term1 - term2 + 1

    return griewank


for _d in pretty_exp(10):
    make_griewank(_d)


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[
        [8.05502, 9.66459],
        [8.05502, -9.66459],
        [-8.05502, 9.66459],
        [-8.05502, -9.66459],
    ],
)
def holder_table(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    x12 = tc.pow(x1, 2.0)
    x22 = tc.pow(x2, 2.0)
    fact1 = tc.sin(x1) * tc.cos(x2)
    fact2 = tc.exp(tc.abs(1 - tc.sqrt(x12 + x22) / tc.pi))
    return -tc.abs(fact1 * fact2)


@torch_problem(
    domain_lower=[0.0, 0.0],
    domain_upper=[10.0, 10.0],
    known_optima=[],
)
def langermann_2d(x: tc.Tensor) -> tc.Tensor:
    cvec = tc.tensor([1.0, 2.0, 5.0, 2.0, 3.0])
    A = tc.tensor(
        [
            [3.0, 5.0],
            [5.0, 2.0],
            [2.0, 1.0],
            [1.0, 4.0],
            [7.0, 9.0],
        ]
    )
    xmat = x[..., None, :]
    inner = tc.sum(tc.pow(xmat - A, 2.0), dim=-1)
    return tc.sum(cvec * tc.exp(-inner / tc.pi) * tc.cos(tc.pi * inner), dim=-1)


def make_levy(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-10.0 for _ in range(d)],
        domain_upper=[10.0 for _ in range(d)],
        known_optima=[[1.0 for _ in range(d)]],
        name=f"levy_{d}d",
    )
    def levy(x: tc.Tensor) -> tc.Tensor:
        w = 1 + (x - 1) / 4

        term1 = tc.pow(tc.sin(tc.pi * w[..., 0]), 2.0)
        term2 = tc.sum(
            tc.pow(w[..., :-1] - 1, 2.0) * (1 + 10 * tc.pow(tc.sin(tc.pi * w[..., :-1] + 1), 2.0)),
            dim=-1,
        )
        term3 = tc.pow(w[..., -1] - 1, 2.0) * (1 + tc.pow(tc.sin(2 * tc.pi * w[..., -1]), 2.0))

        return term1 + term2 + term3

    return levy


for _d in pretty_exp(10):
    make_levy(_d)


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[[1.0, 1.0]],
)
def levy_n13(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(tc.sin(3 * tc.pi * x1), 2.0)
    term2 = tc.pow(x1 - 1, 2.0) * (1 + tc.pow(tc.sin(3 * tc.pi * x2), 2.0))
    term3 = tc.pow(x2 - 1, 2.0) * (1 + tc.pow(tc.sin(2 * tc.pi * x2), 2.0))
    return term1 + term2 + term3


def make_rastrigin(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-5.12 for _ in range(d)],
        domain_upper=[5.12 for _ in range(d)],
        known_optima=[[0.0 for _ in range(d)]],
        name=f"rastrigin_{d}d",
    )
    def rastrigin(x: tc.Tensor) -> tc.Tensor:
        return 10 * d + tc.sum(tc.pow(x, 2.0) - 10 * tc.cos(2 * tc.pi * x), dim=-1)

    return rastrigin


for _d in pretty_exp(10):
    make_rastrigin(_d)


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[[0.0, 0.0]],
)
def schaffer_n2(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]

    fact1 = tc.pow(tc.sin(tc.pow(x1, 2.0) - tc.pow(x2, 2.0)), 2.0) - 0.5
    fact2 = tc.pow(1 + 0.001 * (tc.pow(x1, 2.0) + tc.pow(x2, 2.0)), 2.0)

    return 0.5 + fact1 / fact2


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[],
)
def schaffer_n4(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    fact1 = tc.pow(tc.cos(tc.sin(tc.abs(tc.pow(x1, 2.0) - tc.pow(x2, 2.0)))), 2.0) - 0.5
    fact2 = tc.pow(1 + 0.001 * (tc.pow(x1, 2.0) + tc.pow(x2, 2.0)), 2.0)
    return 0.5 + fact1 / fact2


def make_schwefel(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-500.0 for _ in range(d)],
        domain_upper=[500.0 for _ in range(d)],
        known_optima=[[420.9687 for _ in range(d)]],
        name=f"schwefel_{d}d",
    )
    def schwefel(x: tc.Tensor) -> tc.Tensor:
        eps = 1e-5
        term1 = 418.9829 * tc.tensor(d, dtype=x.dtype)
        term2 = tc.sum(x * tc.sin(tc.sqrt(tc.abs(x) + eps)), dim=-1)
        return term1 - term2

    return schwefel


for _d in pretty_exp(10):
    make_schwefel(_d)


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[],
)
def schubert(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0:1]
    x2 = x[..., 1:2]
    i = tc.arange(5, dtype=x.dtype) + 1
    sum1 = tc.sum(i * tc.cos((i + 1) * x1 + i), dim=-1)
    sum2 = tc.sum(i * tc.cos((i + 1) * x2 + i), dim=-1)
    return sum1 * sum2


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[[0.0, 0.0]],
)
def boachevsky_1(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(x1, 2.0)
    term2 = 2 * tc.pow(x2, 2.0)
    term3 = -0.3 * tc.cos(3 * tc.pi * x1)
    term4 = -0.4 * tc.cos(4 * tc.pi * x2)
    return term1 + term2 + term3 + term4 + 0.7


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[[0.0, 0.0]],
)
def boachevsky_2(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(x1, 2.0)
    term2 = 2 * tc.pow(x2, 2.0)
    term3 = -0.3 * tc.cos(3 * tc.pi * x1) * tc.cos(4 * tc.pi * x2)
    return term1 + term2 + term3 + 0.3


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[[0.0, 0.0]],
)
def boachevsky_3(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(x1, 2.0)
    term2 = 2 * tc.pow(x2, 2.0)
    term3 = -0.3 * tc.cos(3 * tc.pi * x1 + 4 * tc.pi * x2)
    return term1 + term2 + term3 + 0.3


def make_perm_func_0db(d: int) -> Problem:
    @torch_problem(
        domain_lower=[float(-d) for _ in range(d)],
        domain_upper=[float(d) for _ in range(d)],
        known_optima=[[1 / (i + 1) for i in range(d)]],
        name=f"perm_func_0db_{d}d",
    )
    def perm_func_0db(x: tc.Tensor) -> tc.Tensor:
        b = 10.0
        j = tc.arange(d, dtype=x.dtype) + 1
        i = j[:, None]
        inner = tc.sum((j + b) * (tc.pow(x[..., None, :], i) - tc.pow(j, -i)), dim=-1)
        return tc.sum(tc.pow(inner, 2.0), dim=-1)

    return perm_func_0db


for _d in pretty_exp(4):
    make_perm_func_0db(_d)


def make_rotated_hyper_ellipsoid(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-65.536 for _ in range(d)],
        domain_upper=[65.536 for _ in range(d)],
        known_optima=[[0.0 for i in range(d)]],
        name=f"rotated_hyper_ellipsoid_{d}d",
    )
    def rotated_hyper_ellipsoid(x: tc.Tensor) -> tc.Tensor:
        x2 = tc.pow(x, 2.0)
        return tc.sum((d - tc.arange(d, dtype=x.dtype)) * x2, dim=-1)

    return rotated_hyper_ellipsoid


for _d in pretty_exp(10):
    make_rotated_hyper_ellipsoid(_d)


def make_sphere(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-5.12 for _ in range(d)],
        domain_upper=[5.12 for _ in range(d)],
        known_optima=[[0.0 for i in range(d)]],
        name=f"sphere_{d}d",
    )
    def sphere(x: tc.Tensor) -> tc.Tensor:
        return tc.sum(tc.pow(x, 2.0), dim=-1)

    return sphere


for _d in pretty_exp(10):
    make_sphere(_d)


def make_sum_of_different_powers(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-1.0 for _ in range(d)],
        domain_upper=[1.0 for _ in range(d)],
        known_optima=[[0.0 for i in range(d)]],
        name=f"sum_of_different_powers_{d}d",
    )
    def sum_of_different_powers(x: tc.Tensor) -> tc.Tensor:
        powers = tc.arange(d, dtype=x.dtype) + 2
        return tc.sum(tc.pow(tc.abs(x), powers), dim=-1)

    return sum_of_different_powers


for _d in pretty_exp(10):
    make_sum_of_different_powers(_d)


def make_sum_squares(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-10.0 for _ in range(d)],
        domain_upper=[10.0 for _ in range(d)],
        known_optima=[[0.0 for i in range(d)]],
        name=f"sum_squares_{d}d",
    )
    def sum_squares(x: tc.Tensor) -> tc.Tensor:
        terms = (tc.arange(d, dtype=x.dtype) + 1) * tc.pow(x, 2.0)
        return tc.sum(terms, dim=-1)

    return sum_squares


for _d in pretty_exp(10):
    make_sum_squares(_d)


def make_trid(d: int) -> Problem:
    @torch_problem(
        domain_lower=[float(d * d) for _ in range(d)],
        domain_upper=[float(d * d) for _ in range(d)],
        known_optima=[[float((i + 1) * (d - i)) for i in range(d)]],
        name=f"trid_{d}d",
    )
    def trid(x: tc.Tensor) -> tc.Tensor:
        term1 = tc.sum(tc.pow(x - 1, 2.0), dim=-1)
        term2 = tc.sum(x[..., 1:] * x[..., :-1], dim=-1)
        return term1 - term2

    return trid


for _d in pretty_exp(10):
    make_trid(_d)


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[[1.0, 3.0]],
)
def booth(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(x1 + 2 * x2 - 7, 2.0)
    term2 = tc.pow(2 * x1 + x2 - 5, 2.0)
    return term1 + term2


@torch_problem(
    domain_lower=[-10.0, -10.0],
    domain_upper=[10.0, 10.0],
    known_optima=[[0.0, 0.0]],
)
def matyas(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = 0.26 * (tc.pow(x1, 2.0) + tc.pow(x2, 2.0))
    term2 = 0.48 * x1 * x2
    return term1 + term2


@torch_problem(
    domain_lower=[-1.5, -3.0],
    domain_upper=[4.0, 4.0],
    known_optima=[[-0.54719, -1.54719]],
)
def mccormick(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.sin(x1 + x2)
    term2 = tc.pow(x1 - x2, 2.0)
    term3 = -1.5 * x1
    term4 = 2.5 * x2
    return term1 + term2 + term3 + term4 + 1


@torch_problem(
    domain_lower=[0.0, 0.0, 0.0, 0.0],
    domain_upper=[4.0, 4.0, 4.0, 4.0],
    known_optima=[],
)
def power_sum_4d(x: tc.Tensor) -> tc.Tensor:
    b = tc.tensor([8, 18, 44, 114], dtype=x.dtype)
    powers = tc.arange(4, dtype=x.dtype) + 1
    inner = tc.sum(x[..., None, :] * powers[:, None], dim=-1)
    return tc.sum(tc.pow(inner - b, 2.0), dim=-1)


def make_zakharov(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-5.0 for _ in range(d)],
        domain_upper=[10.0 for _ in range(d)],
        known_optima=[[0.0 for _ in range(d)]],
        name=f"zakharov_{d}d",
    )
    def zakharov(x: tc.Tensor) -> tc.Tensor:
        i = tc.arange(d, dtype=x.dtype) + 1
        term1 = tc.sum(tc.pow(x, 2.0), dim=-1)
        term2 = tc.pow(tc.sum(0.5 * i * x, dim=-1), 2.0)
        term3 = tc.pow(tc.sum(0.5 * i * x, dim=-1), 4.0)
        return term1 + term2 + term3

    return zakharov


for _d in pretty_exp(10):
    make_zakharov(_d)


@torch_problem(
    domain_lower=[-5.0, -5.0],
    domain_upper=[5.0, 5.0],
    known_optima=[[0.0, 0.0]],
)
def three_hump_camel(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = 2 * tc.pow(x1, 2.0)
    term2 = 1.05 * tc.pow(x1, 4.0)
    term3 = tc.pow(x1, 6.0) / 6.0
    term4 = x1 * x2
    term5 = tc.pow(x2, 2.0)
    return term1 - term2 + term3 + term4 + term5


@torch_problem(
    domain_lower=[-3.0, -2.0],
    domain_upper=[3.0, 2.0],
    known_optima=[
        [0.0898, -0.7126],
        [-0.0898, 0.7126],
    ],
)
def six_hump_camel(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    four = tc.tensor(4.0, dtype=x.dtype)
    term1 = (four - 2.1 * tc.pow(x1, 2.0) + tc.pow(x1, 4.0) / 3.0) * tc.pow(x1, 2.0)
    term2 = x1 * x2
    term3 = (-4 + 4 * tc.pow(x2, 2.0)) * tc.pow(x2, 2.0)
    return term1 + term2 + term3


def make_dixon_price(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-10.0 for _ in range(d)],
        domain_upper=[10.0 for _ in range(d)],
        known_optima=[],  # TODO
        name=f"dixon_price_{d}d",
    )
    def dixon_price(x: tc.Tensor) -> tc.Tensor:
        i = tc.arange(d - 1, dtype=x.dtype) + 2
        x1 = x[..., 0]
        xi = x[..., 1:]
        xi1 = x[..., :-1]
        term1 = tc.pow(x1 - 1, 2.0)
        term2 = tc.sum(i * tc.pow(2 * tc.pow(xi, 2.0) - xi1, 2.0), dim=-1)
        return term1 + term2

    return dixon_price


for _d in pretty_exp(10):
    make_dixon_price(_d)


def make_rosenbrock(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-5.0 for _ in range(d)],
        domain_upper=[10.0 for _ in range(d)],
        known_optima=[[1.0 for _ in range(d)]],
        name=f"rosenbrock_{d}d",
    )
    def rosenbrock(x: tc.Tensor) -> tc.Tensor:
        xp = x[..., :-1]
        xn = x[..., 1:]
        term1 = 100 * tc.pow(xn - tc.pow(xp, 2.0), 2.0)
        term2 = tc.pow(xp - 1, 2.0)
        return tc.sum(term1 + term2, dim=-1)

    return rosenbrock


for _d in pretty_exp(10):
    make_rosenbrock(_d)


@torch_problem(
    domain_lower=[-65.536, -65.536],
    domain_upper=[65.536, 65.536],
    known_optima=[],
)
def de_jong(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    a1 = tc.tensor(
        [
            -32,
            -16,
            0,
            16,
            32,
            -32,
            -16,
            0,
            16,
            32,
            -32,
            -16,
            0,
            16,
            32,
            -32,
            -16,
            0,
            16,
            32,
            -32,
            -16,
            0,
            16,
            32,
        ],
        dtype=x.dtype,
    )
    a2 = tc.tensor(
        [
            -32,
            -32,
            -32,
            -32,
            -32,
            -16,
            -16,
            -16,
            -16,
            -16,
            0,
            0,
            0,
            0,
            0,
            16,
            16,
            16,
            16,
            16,
            32,
            32,
            32,
            32,
            32,
        ],
        dtype=x.dtype,
    )
    i = tc.arange(25, dtype=x.dtype) + 1
    div1 = i + tc.pow(x1[..., None] * a1, 6.0) + tc.pow(x2[..., None] * a2, 6.0)
    div2 = 0.002 + tc.sum(tc.pow(div1, -1.0), dim=-1)
    return tc.pow(div2, -1.0)


@torch_problem(
    domain_lower=[-100.0, -100.0],
    domain_upper=[100.0, 100.0],
    known_optima=[[pi, pi]],
)
def easom(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]

    fact1 = -tc.cos(x1) * tc.cos(x2)
    fact2 = tc.exp(-tc.pow(x1 - tc.pi, 2.0) - tc.pow(x2 - tc.pi, 2.0))

    return fact1 * fact2


# This seems very numerically unstable...
#
# def make_michalewicz(d: int) -> Problem:
#     @torch_problem(
#         domain_lower=[0.0 for _ in range(d)],
#         domain_upper=[pi for _ in range(d)],
#         known_optima=[],
#         name=f"michalewicz_{d}d",
#     )
#     def michalewicz(x: tc.Tensor) -> tc.Tensor:
#         i = tc.arange(d, dtype=x.dtype) + 1
#         return -tc.sum(tc.sin(x) * tc.pow(tc.sin(i * tc.pow(x, 2.0) / tc.pi), 10.0), dim=-1)
#
#     return michalewicz
#
#
# for _d in pretty_exp(10):
#     make_michalewicz(_d)


@torch_problem(
    domain_lower=[-4.5, -4.5],
    domain_upper=[4.5, 4.5],
    known_optima=[[3.0, 0.5]],
)
def beale(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = tc.pow(1.5 - x1 + x1 * x2, 2.0)
    term2 = tc.pow(2.25 - x1 + x1 * tc.pow(x2, 2.0), 2.0)
    term3 = tc.pow(2.625 - x1 + x1 * tc.pow(x2, 3.0), 2.0)
    return term1 + term2 + term3


@torch_problem(
    domain_lower=[-5.0, 0.0],
    domain_upper=[10.0, 15.0],
    known_optima=[[-pi, 12.275], [pi, 2.275], [9.42478, 2.475]],
)
def branin(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    t = 1 / (8 * pi)
    s = 10
    r = 6
    c = 5 / pi
    b = 5.1 / (4 * (pi**2.0))
    a = 1
    term1 = a * tc.pow(x2 - b * tc.pow(x1, 2.0) + c * x1 - r, 2.0)
    term2 = s * (1 - t) * tc.cos(x1)
    return term1 + term2 + s


@torch_problem(
    domain_lower=[-10.0, -10.0, -10.0, -10.0],
    domain_upper=[10.0, 10.0, 10.0, 10.0],
    known_optima=[[1.0, 1.0, 1.0, 1.0]],
)
def colville(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    x4 = x[..., 3]
    term1 = 100 * tc.pow(tc.pow(x1, 2.0) - x2, 2.0)
    term2 = tc.pow(x1 - 1, 2.0)
    term3 = tc.pow(x3 - 1, 2.0)
    term4 = 90 * tc.pow(tc.pow(x3, 2.0) - x4, 2.0)
    term5 = 10.1 * (tc.pow(x2 - 1, 2.0) + tc.pow(x4 - 1, 2.0))
    term6 = 19.8 * (x2 - 1) * (x4 - 1)
    return term1 + term2 + term3 + term4 + term5 + term6


@torch_problem(
    domain_lower=[0.0],
    domain_upper=[1.0],
    known_optima=[],
)
def forrester_et_al_2008(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    fact1 = tc.pow(6 * x1 - 2, 2.0)
    fact2 = tc.sin(12 * x1 - 4)
    return fact1 * fact2


@torch_problem(
    domain_lower=[-2.0, -2.0],
    domain_upper=[2.0, 2.0],
    known_optima=[],
)
def goldstein_price(x: tc.Tensor) -> tc.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    fact1a = tc.pow(x1 + x2 + 1.0, 2.0)
    fact1b: tc.Tensor = (
        19.0 - 14.0 * x1 + 3.0 * tc.pow(x1, 2.0) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * tc.pow(x2, 2.0)
    )
    fact1 = 1.0 + fact1a * fact1b
    fact2a = tc.pow(2.0 * x1 - 3.0 * x2, 2.0)
    fact2b: tc.Tensor = (
        18.0
        - 32.0 * x1
        + 12.0 * tc.pow(x1, 2.0)
        + 48.0 * x2
        - 36.0 * x1 * x2
        + 27.0 * tc.pow(x2, 2.0)
    )
    fact2: tc.Tensor = 30.0 + fact2a * fact2b
    return fact1 * fact2


@check_shapes(
    "alpha: [inner]",
    "A: [inner, input]",
    "P: [inner, input]",
    "known_optima: [n_optima, input]",
)
def make_hartmann(
    name: str,
    alpha: AnyNDArray,
    A: AnyNDArray,
    P: AnyNDArray,
    known_optima: Collection[AnyNDArray | Sequence[float]] = (),
) -> Problem:
    _, d = A.shape

    @torch_problem(
        domain_lower=[0.0 for _ in range(d)],
        domain_upper=[1.0 for _ in range(d)],
        known_optima=known_optima,
        name=name,
    )
    def hartmann(x: tc.Tensor) -> tc.Tensor:
        alpha_ = tc.tensor(alpha, dtype=x.dtype)
        A_ = tc.tensor(A, dtype=x.dtype)
        P_ = tc.tensor(P, dtype=x.dtype)
        inner = tc.sum(A_ * tc.pow((x[..., None, :] - P_), 2.0), dim=-1)
        outer = tc.sum(alpha_ * tc.exp(-inner), dim=-1)
        return -outer

    return hartmann


hartmann_3d = make_hartmann(
    "hartmann_3d",
    alpha=np.array([1.0, 1.2, 3.0, 3.2]),
    A=np.array(
        [
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35],
        ]
    ),
    P=(10**-4)
    * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ],
    ),
    known_optima=[[0.114614, 0.555649, 0.852549]],
)


hartmann_4d = make_hartmann(
    "hartmann_4d",
    alpha=np.array([1.0, 1.2, 3.0, 3.2]),
    A=np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
    ),
    P=(10**-4)
    * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
    ),
)


hartmann_6d = make_hartmann(
    "hartmann_6d",
    alpha=np.array([1.0, 1.2, 3.0, 3.2]),
    A=np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
    ),
    P=(10**-4)
    * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
    ),
)


def make_perm_func_db(d: int) -> Problem:
    @torch_problem(
        domain_lower=[float(-d) for _ in range(d)],
        domain_upper=[float(d) for _ in range(d)],
        known_optima=[[float(i + 1) for i in range(d)]],
        name=f"perm_func_db_{d}d",
    )
    def perm_func_db(x: tc.Tensor) -> tc.Tensor:
        b = 10.0
        j = tc.arange(d, dtype=x.dtype) + 1
        i = j[:, None]
        inner = tc.sum((tc.pow(j, i) + b) * (tc.pow(x[..., None, :] / j, i) - 1), dim=-1)
        return tc.sum(tc.pow(inner, 2.0), dim=-1)

    return perm_func_db


for _d in pretty_exp(4):
    make_perm_func_db(_d)


def make_powell(d: int) -> Problem:
    assert d % 4 == 0, d

    @torch_problem(
        domain_lower=[-4.0 for _ in range(d)],
        domain_upper=[5.0 for _ in range(d)],
        known_optima=[[0.0 for i in range(d)]],
        name=f"powell_{d}d",
    )
    def powell(x: tc.Tensor) -> tc.Tensor:
        x = tc.reshape(x, x.shape[:-1] + (-1, 4))
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]
        x4 = x[..., 3]
        term1 = tc.pow(x1 + 10 * x2, 2.0)
        term2 = 5 * tc.pow(x3 - x4, 2.0)
        term3 = tc.pow(x2 - 2 * x3, 4.0)
        term4 = 10 * tc.pow(x1 - x4, 4.0)
        return tc.sum(term1 + term2 + term3 + term4, dim=-1)

    return powell


for _d in pretty_exp(9):
    make_powell(4 * _d)


@torch_problem(
    domain_lower=[0.0, 0.0, 0.0, 0.0],
    domain_upper=[10.0, 10.0, 10.0, 10.0],
    known_optima=[[4.0, 4.0, 4.0, 4.0]],
)
def shekel(x: tc.Tensor) -> tc.Tensor:
    b = 0.1 * tc.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=x.dtype)
    C = tc.tensor(
        [
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        ],
        dtype=x.dtype,
    )

    inner = tc.sum(tc.pow(x[..., None] - C, 2.0), dim=-2)
    outer = tc.sum(1.0 / (inner + b), dim=-1)
    return -outer


def make_styblinski_tang(d: int) -> Problem:
    @torch_problem(
        domain_lower=[-5.0 for _ in range(d)],
        domain_upper=[5.0 for _ in range(d)],
        known_optima=[[-2.903534 for i in range(d)]],
        name=f"styblinski_tang_{d}d",
    )
    def styblinski_tang(x: tc.Tensor) -> tc.Tensor:
        return 0.5 * tc.sum(tc.pow(x, 4.0) - 16.0 * tc.pow(x, 2.0) + 5.0 * x, dim=-1)

    return styblinski_tang


for _d in pretty_exp(10):
    make_styblinski_tang(_d)

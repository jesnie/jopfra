import datetime as dt
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as tc
from check_shapes import check_shapes

from jopfra.paths import MatplotlibPngFile, MiscDir
from jopfra.problems.api import Problem, torch_problem


@check_shapes(
    "return[0]: [n_data]",
    "return[1]: [n_data]",
    "return[2]: [n_quantiles]",
    "return[3]: [n_quantiles, degree]",
)
def _load_data() -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
    root = Path(__file__).parent

    due_date = dt.datetime(2023, 11, 7)
    conception_date = due_date - dt.timedelta(weeks=40)

    size_df = pd.read_csv(root / "size.csv")
    size_df["date"] = pd.to_datetime(size_df["date"])
    size_df["age_weeks"] = (size_df.date - conception_date) / dt.timedelta(
        weeks=1
    )  # type: ignore[operator]
    size_age_weeks = tc.as_tensor(size_df.age_weeks)
    size_weight_g = tc.as_tensor(size_df.weight_g)

    coeffs_df = pd.read_csv(root / "coefficientsGlobalV3.csv").set_index(["fetalDimension"])
    coeffs = tc.tensor(coeffs_df.loc["EFW"].values)
    coeffs_index = coeffs[:, 0]
    coeffs = coeffs[:, 1:]

    return size_age_weeks, size_weight_g, coeffs_index, coeffs


_size_age_weeks, _size_weight_g, _coeffs_index, _coeffs = _load_data()


@check_shapes(
    "c: [broadcast batch..., degree]",
    "x: [broadcast batch...]",
    "return: [batch...]",
)
def _polynomial(c: tc.Tensor, x: tc.Tensor) -> tc.Tensor:
    degree = c.shape[-1]
    pows = tc.pow(x[..., None], tc.arange(degree, dtype=x.dtype))
    return tc.sum(c * pows, dim=-1)


@check_shapes(
    "age_weeks: [broadcast batch...]",
    "quantile: [broadcast batch...]",
    "return: [batch...]",
)
def _fetal_growth(age_weeks: tc.Tensor, quantile: tc.Tensor) -> tc.Tensor:
    i1 = tc.searchsorted(_coeffs_index, quantile)
    i1 = tc.minimum(tc.maximum(i1, tc.tensor(1)), tc.tensor(len(_coeffs_index) - 1))
    i0 = i1 - 1
    q0 = _coeffs_index[i0]
    q1 = _coeffs_index[i1]
    logy0 = _polynomial(_coeffs[i0, :], age_weeks)
    logy1 = _polynomial(_coeffs[i1, :], age_weeks)
    logy = logy0 + (quantile - q0) * (logy1 - logy0) / (q1 - q0)
    return tc.exp(logy)


def _loss(x: tc.Tensor) -> tc.Tensor:
    translate = x[..., 0, None]
    quantile = x[..., 1, None]
    std = x[..., 2, None]
    dist = tc.distributions.normal.Normal(0.0, std)  # type: ignore[no-untyped-call]
    err = _fetal_growth(_size_age_weeks + translate, quantile) - _size_weight_g
    return -tc.sum(dist.log_prob(err), dim=-1)  # type: ignore[no-untyped-call]


def _plot(dest: MiscDir, x: tc.Tensor) -> None:
    translate = x[0]
    quantile = x[1]
    std = x[2]

    loss = _loss(x)
    age_weeks = tc.linspace(25.0, 40.0, 200)
    y_median = _fetal_growth(age_weeks, tc.tensor(0.5))
    y_fitted = _fetal_growth(age_weeks + translate, quantile)

    png = dest.get("growth.png", MatplotlibPngFile)
    with png.subplots(1, 1, figsize=(12, 16)) as (_, ax):
        ax.set_title(f"L: {loss:.4}; T: {translate*7:.3}; Q: {quantile:.2%}; std: {std:.3}")
        ax.plot(age_weeks, y_median, label="Median")
        ax.plot(age_weeks, y_fitted, label="Fitted")
        ax.scatter(_size_age_weeks, _size_weight_g, label="Data")
        ax.legend()


torch_problem(
    domain_lower=[-3.0, 0.01, 1e-2],
    domain_upper=[3.0, 0.99, 1e2],
    known_optima=[],
    name="fetal_growth",
    plot=_plot,
)(_loss)

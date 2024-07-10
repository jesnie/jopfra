from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeAlias

import pandas as pd
import torch as tc
from check_shapes import check_shapes

ROOT = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Dataset:
    name: str
    x: tc.Tensor
    y: tc.Tensor

    @check_shapes(
        "self.x: [n_rows, n_inputs]",
        "self.y: [n_rows, n_outputs]",
    )
    def __post_init__(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_inputs(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.y.shape[1])

    def __len__(self) -> int:
        return self.n_rows


DatasetFactory: TypeAlias = Callable[[], Dataset]


@lru_cache(maxsize=1)
def get_dataset(factory: DatasetFactory) -> Dataset:
    return factory()


def sin_dataset() -> Dataset:
    rng = tc.Generator().manual_seed(20231202)
    x = 10.0 * tc.rand((25, 1), generator=rng)
    y = tc.sin(x)
    return Dataset("sin", x, y)


def fetal_size_dataset() -> Dataset:
    # pylint: disable=import-outside-toplevel
    from jopfra.problems.fetal_growth import _size_age_weeks, _size_weight_g

    x = _size_age_weeks[:, None]
    y = _size_weight_g[:, None]
    return Dataset("fetal_size", x, y)


def baby_size_dataset() -> Dataset:
    df = pd.read_csv(ROOT / "baby_size.csv")
    x = tc.as_tensor(df.age_days)[:, None]
    y = tc.as_tensor(df.weight_g)[:, None]
    return Dataset("baby_size", x, y)


def mauna_loa_co2_dataset() -> Dataset:
    df = pd.read_csv(ROOT / "co2_mm_mlo.csv", comment="#")
    x = tc.as_tensor(df["decimal date"])[:, None]
    y = tc.as_tensor(df["average"])[:, None]
    return Dataset("mauna_loa_co2", x, y)


def gpflow_example_1_dataset() -> Dataset:
    x = tc.as_tensor(
        [
            [0.865],
            [0.666],
            [0.804],
            [0.771],
            [0.147],
            [0.866],
            [0.007],
            [0.026],
            [0.171],
            [0.889],
            [0.243],
            [0.028],
        ]
    )
    y = tc.as_tensor(
        [
            [1.57],
            [3.48],
            [3.12],
            [3.91],
            [3.07],
            [1.35],
            [3.80],
            [3.82],
            [3.49],
            [1.30],
            [4.00],
            [3.82],
        ]
    )
    return Dataset("gpflow_example_1", x, y)


def gpflow_example_2_dataset() -> Dataset:
    x = tc.as_tensor([[-0.5], [0.0], [0.4], [0.5]])
    y = tc.as_tensor([[1.0], [0.0], [0.6], [0.4]])
    return Dataset("gpflow_example_2", x, y)


def gpflow_example_3_dataset() -> Dataset:
    x = tc.as_tensor(
        [
            [-0.4, -0.5],
            [0.1, -0.3],
            [0.4, -0.4],
            [0.5, -0.5],
            [-0.5, 0.3],
            [0.0, 0.5],
            [0.4, 0.4],
            [0.5, 0.3],
        ]
    )
    y = tc.as_tensor([[0.8], [0.0], [0.5], [0.3], [1.0], [0.2], [0.7], [0.5]])
    return Dataset("gpflow_example_3", x, y)


def gpflow_example_4_dataset() -> Dataset:
    # Quadtratic mean function
    x = tc.as_tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = tc.as_tensor([[2.0], [1.7], [1.6], [1.7], [2.0]])
    return Dataset("gpflow_example_4", x, y)


def gpflow_example_5_dataset() -> Dataset:
    # Linear mean function
    x = tc.as_tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = tc.as_tensor([[1.0], [1.3], [1.2], [1.5], [1.4]])
    return Dataset("gpflow_example_5", x, y)


def gpflow_example_6_dataset() -> Dataset:
    # Deal with outlier
    x = tc.as_tensor(
        [[0.177], [0.183], [0.428], [0.838], [0.827], [0.293], [0.270], [0.593], [0.031], [0.650]]
    )
    y = tc.as_tensor(
        [[1.22], [1.17], [1.99], [2.29], [2.29], [1.28], [1.20], [1.82], [1.01], [1.93]]
    )
    return Dataset("gpflow_example_6", x, y)

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, TypeAlias

import torch as tc
from check_shapes import check_shapes


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

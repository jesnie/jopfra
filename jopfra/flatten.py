from typing import overload

import numpy as np
import torch as tc
from check_shapes import check_shapes

from jopfra.problems.api import Evaluation
from jopfra.types import AnyNDArray


class Flattener:
    def __init__(self, batch_shape: tuple[int, ...]) -> None:
        self._batch_shape = batch_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def dim(self) -> int:
        return len(self._batch_shape)

    @property
    def size(self) -> int:
        return int(np.prod(self._batch_shape))

    @overload
    def flatten(self, y: AnyNDArray) -> AnyNDArray:
        ...

    @overload
    def flatten(self, y: tc.Tensor) -> tc.Tensor:
        ...

    @overload
    def flatten(self, y: Evaluation) -> Evaluation:
        ...

    @check_shapes(
        "y: [batch_shape..., item_shape...]",
        "return: [prod_batch_shape, item_shape...]",
    )
    def flatten(
        self, y: AnyNDArray | tc.Tensor | Evaluation
    ) -> AnyNDArray | tc.Tensor | Evaluation:
        if isinstance(y, Evaluation):
            return Evaluation(
                y.problem, self.flatten(y.x), self.flatten(y.loss), self.flatten(y.grads)
            )

        if isinstance(y, tc.Tensor):
            assert self.shape == y.shape[: self.dim], (self.shape, y.shape)
            return tc.reshape(y, (self.size,) + y.shape[self.dim :])

        assert isinstance(y, np.ndarray), y
        assert self.shape == y.shape[: self.dim], (self.shape, y.shape)
        return np.reshape(y, (self.size,) + y.shape[self.dim :])

    @overload
    def unflatten(self, y: AnyNDArray) -> AnyNDArray:
        ...

    @overload
    def unflatten(self, y: tc.Tensor) -> tc.Tensor:
        ...

    @overload
    def unflatten(self, y: Evaluation) -> Evaluation:
        ...

    @check_shapes(
        "y: [prod_batch_shape, item_shape...]",
        "return: [batch_shape..., item_shape...]",
    )
    def unflatten(
        self, y: AnyNDArray | tc.Tensor | Evaluation
    ) -> AnyNDArray | tc.Tensor | Evaluation:
        if isinstance(y, Evaluation):
            return Evaluation(
                y.problem, self.unflatten(y.x), self.unflatten(y.loss), self.unflatten(y.grads)
            )

        if isinstance(y, tc.Tensor):
            assert self.size == y.shape[0]
            return tc.reshape(y, self.shape + y.shape[1:])

        assert isinstance(y, np.ndarray), y
        assert self.size == y.shape[0]
        return np.reshape(y, self.shape + y.shape[1:])

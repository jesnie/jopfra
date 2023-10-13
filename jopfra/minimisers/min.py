from collections.abc import Iterator
from typing import Final

import numpy as np

from jopfra.flatten import Flattener
from jopfra.minimisers.api import IterMinimiser, SingleMinimiser, iter_minimisers
from jopfra.minimisers.random import random
from jopfra.minimisers.sobol import sobol
from jopfra.problems.api import Evaluation, Problem


class SingleMin(SingleMinimiser):
    def __init__(self, source: SingleMinimiser, scale: int) -> None:
        self._source = source
        self._scale = scale

    def single_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Evaluation:
        flat = Flattener(batch_shape)
        flat_y = self._source.single_minimise(problem, (flat.size, self._scale))
        assert (flat.size, self._scale) == flat_y.shape, (flat.size, self._scale, flat_y.shape)
        idx = np.argmin(flat_y.loss, axis=1)
        assert (flat.size,) == idx.shape, (flat.size, idx.shape)
        flat_y = Evaluation(
            flat_y.problem,
            x=np.squeeze(np.take_along_axis(flat_y.x, idx[:, None, None], axis=1), axis=1),
            loss=np.squeeze(np.take_along_axis(flat_y.loss, idx[:, None], axis=1), axis=1),
            grads=np.squeeze(np.take_along_axis(flat_y.grads, idx[:, None, None], axis=1), axis=1),
        )
        assert (flat.size,) == flat_y.shape, (flat.size, flat_y.shape)
        return flat.unflatten(flat_y)


sobol_min_init: Final[SingleMin] = SingleMin(sobol, scale=100)
random_min_init: Final[SingleMin] = SingleMin(random, scale=100)


class IterMin(IterMinimiser):
    def __init__(self, source: IterMinimiser, scale: int, memory: bool) -> None:
        self._source = source
        self._scale = scale
        self._memory = memory

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        flat = Flattener(batch_shape)
        flat_prev: Evaluation | None = None
        for flat_y in self._source.iter_minimise(problem, (flat.size, self._scale)):
            assert (flat.size, self._scale) == flat_y.shape, (flat.size, self._scale, flat_y.shape)
            idx = np.argmin(flat_y.loss, axis=1)
            assert (flat.size,) == idx.shape, (flat.size, idx.shape)
            flat_y = Evaluation(
                flat_y.problem,
                x=np.squeeze(np.take_along_axis(flat_y.x, idx[:, None, None], axis=1), axis=1),
                loss=np.squeeze(np.take_along_axis(flat_y.loss, idx[:, None], axis=1), axis=1),
                grads=np.squeeze(
                    np.take_along_axis(flat_y.grads, idx[:, None, None], axis=1), axis=1
                ),
            )
            assert (flat.size,) == flat_y.shape, (flat.size, flat_y.shape)
            if self._memory:
                if flat_prev is not None:
                    new_better = flat_y.loss < flat_prev.loss
                    flat_y = Evaluation(
                        flat_y.problem,
                        np.where(new_better, flat_y.x, flat_prev.x),
                        np.where(new_better, flat_y.loss, flat_prev.loss),
                        np.where(new_better, flat_y.grads, flat_prev.grads),
                    )
                flat_prev = flat_y
            yield flat.unflatten(flat_y)


sobol_min: Final[IterMin] = IterMin(sobol, scale=1, memory=True)
random_min: Final[IterMin] = IterMin(random, scale=1, memory=True)

iter_minimisers["sobol_min"] = sobol_min
iter_minimisers["random_min"] = random_min

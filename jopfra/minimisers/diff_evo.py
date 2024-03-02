from collections.abc import Iterator
from typing import Callable

import numpy as np
from check_shapes import check_shapes

from jopfra.api import Evaluation, Problem
from jopfra.flatten import Flattener
from jopfra.minimisers.api import IterMinimiser, SingleMinimiser, iter_minimisers
from jopfra.minimisers.sobol import sobol
from jopfra.problems.utils import wrap_domain
from jopfra.types import AnyNDArray


@check_shapes(
    "loss: [batch..., n]",
    "return: [batch..., n]",
)
def softmax(loss: AnyNDArray) -> AnyNDArray:
    return np.exp(loss - np.logaddexp.reduce(loss, axis=-1))  # type: ignore[no-any-return]


@check_shapes(
    "loss: [batch..., n]",
    "return: [batch..., n]",
)
def softminsub(loss: AnyNDArray) -> AnyNDArray:
    return softmax(-loss)


@check_shapes(
    "loss: [batch..., n]",
    "return: [batch..., n]",
)
def softmindiv(loss: AnyNDArray) -> AnyNDArray:
    return softmax(1 / loss)


@check_shapes(
    "loss: [batch..., n]",
    "return: [batch..., n]",
)
def besthalf(loss: AnyNDArray) -> AnyNDArray:
    n = loss.shape[-1]
    k = n // 2
    assert k > 0, loss.shape
    idx = np.argpartition(loss, k, axis=-1)[:, :k]
    p = np.zeros_like(loss)
    np.put_along_axis(p, idx, 1 / k, axis=-1)
    return p


class DiffEvo(IterMinimiser):
    def __init__(
        self,
        initial_value: SingleMinimiser,
        population_size: int,
        select: Callable[[AnyNDArray], AnyNDArray],
        seed: int | None = 20230907,
    ) -> None:
        assert population_size > 3, population_size
        self._initial_value = initial_value
        self._population_size = population_size
        self._select = select
        self._seed = seed

    def iter_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        rng = np.random.default_rng(self._seed)
        flat = Flattener(batch_shape)

        flat_y = self._initial_value.single_minimise(problem, (flat.size, self._population_size))
        input_dim = flat_y.x.shape[-1]

        while True:
            best_idx = np.argmin(flat_y.loss, axis=1)
            best_y = Evaluation(
                flat_y.problem,
                x=np.squeeze(np.take_along_axis(flat_y.x, best_idx[:, None, None], axis=1), axis=1),
                loss=np.squeeze(np.take_along_axis(flat_y.loss, best_idx[:, None], axis=1), axis=1),
                grads=np.squeeze(
                    np.take_along_axis(flat_y.grads, best_idx[:, None, None], axis=1), axis=1
                ),
            )
            yield flat.unflatten(best_y)

            p = self._select(flat_y.loss)
            new_x = []
            for i in range(flat.size):
                new_xi = []
                for _ in range(self._population_size):
                    p0, p1, p2 = rng.choice(self._population_size, 3, replace=False, p=p[i])
                    flat_x = flat_y.x[i, p0] + rng.random(input_dim) * (
                        flat_y.x[i, p1] - flat_y.x[i, p2]
                    )
                    flat_x = wrap_domain(problem, flat_x)
                    new_xi.append(flat_x)
                new_x.append(new_xi)

            flat_y = problem(np.array(new_x))


iter_minimisers["diff_evo_softminsub"] = DiffEvo(sobol, 20, softminsub)
iter_minimisers["diff_evo_softmindiv"] = DiffEvo(sobol, 20, softmindiv)
iter_minimisers["diff_evo_besthalf"] = DiffEvo(sobol, 20, besthalf)

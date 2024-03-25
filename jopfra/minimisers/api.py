from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Callable, TypeAlias

from jopfra.problems.api import Evaluation, Problem


class SingleMinimiser(ABC):
    @abstractmethod
    def single_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Evaluation: ...


single_minimisers: dict[str, SingleMinimiser] = {}


Stop: TypeAlias = Callable[[Evaluation], bool]


class StoppingCriteria(ABC):
    @abstractmethod
    def stop(self) -> Stop: ...


class IterMinimiserAdapter(SingleMinimiser):
    def __init__(self, minimiser: IterMinimiser, criteria: StoppingCriteria) -> None:
        self._minimiser = minimiser
        self._criteria = criteria

    def single_minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Evaluation:
        stop = self._criteria.stop()
        for y in self._minimiser.iter_minimise(problem, batch_shape):
            if stop(y):
                break
        else:
            raise AssertionError(f"{self._criteria}.iter_minimise should loop forever.")
        assert y is not None
        return y


class IterMinimiser(ABC):
    @abstractmethod
    def iter_minimise(
        self, problem: Problem, batch_shape: tuple[int, ...]
    ) -> Iterator[Evaluation]: ...

    def to_single(self, criteria: StoppingCriteria) -> IterMinimiserAdapter:
        return IterMinimiserAdapter(self, criteria)


iter_minimisers: dict[str, IterMinimiser] = {}

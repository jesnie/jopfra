from abc import ABC, abstractmethod
from collections.abc import Iterator

from jopfra.problems.api import Evaluation, Problem


class Minimiser(ABC):
    @abstractmethod
    def minimise(self, problem: Problem, batch_shape: tuple[int, ...]) -> Iterator[Evaluation]:
        ...


minimisers: dict[str, Minimiser] = {}

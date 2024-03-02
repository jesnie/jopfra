from jopfra.api import Evaluation
from jopfra.minimisers.api import Stop, StoppingCriteria


class CallCountStoppingCriteria(StoppingCriteria):
    def __init__(self, max_calls: int) -> None:
        assert max_calls >= 0, max_calls
        self._max_calls = max_calls

    def stop(self) -> Stop:
        n_calls = 0

        def _stop(y: Evaluation) -> bool:
            nonlocal n_calls
            n_calls += 1
            return n_calls >= self._max_calls

        return _stop

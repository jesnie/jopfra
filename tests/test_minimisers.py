import numpy as np
import pytest

from jopfra.api import Evaluation
from jopfra.minimisers.api import Stop, StoppingCriteria, iter_minimisers
from jopfra.minimisers.stopping_criteria import CallCountStoppingCriteria
from jopfra.problems.quadratic import make_quadratic


@pytest.mark.parametrize("minimiser_name", iter_minimisers)
def test_minimiser__batch_shape(minimiser_name: str) -> None:
    problem = make_quadratic("test_problem", [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    minimiser = iter_minimisers[minimiser_name]

    y = minimiser.to_single(CallCountStoppingCriteria(1)).single_minimise(problem, (4, 2))

    assert (4, 2) == y.shape
    assert 3 == y.n_inputs
    assert (4, 2, 3) == y.x.shape


@pytest.mark.parametrize("minimiser_name", iter_minimisers)
def test_minimiser__find_minimum(minimiser_name: str) -> None:
    minimum = [0.2, 0.8, 0.5]
    problem = make_quadratic("test_problem", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], minimum)
    minimiser = iter_minimisers[minimiser_name]

    class _StoppingCriteria(StoppingCriteria):
        def stop(self) -> Stop:
            n_calls = 0

            def _stop(y: Evaluation) -> bool:
                nonlocal n_calls
                n_calls += 1
                assert n_calls < 10_000, n_calls
                max_dist = np.max(np.abs(y.x - minimum))
                result: bool = max_dist < 0.05
                return result

            return _stop

    minimiser.to_single(_StoppingCriteria()).single_minimise(problem, ())

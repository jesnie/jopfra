from collections.abc import Iterator

import numpy as np
import pytest

from jopfra.problems.api import problems


@pytest.fixture(autouse=True)
def suppress_np_scientific_notation() -> Iterator[None]:
    with np.printoptions(suppress=True):
        yield


@pytest.fixture(autouse=True)
def reset_problems() -> Iterator[None]:
    old_problems = dict(problems)
    yield
    problems.clear()
    problems.update(old_problems)

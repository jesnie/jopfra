from collections.abc import Sequence
from typing import Callable

import numpy as np
import torch as tc
from check_shapes import check_shapes
from torch import nn
from torch.nn import functional as F

from jopfra.flatten import Flattener
from jopfra.problems.api import Problem, problem
from jopfra.types import AnyNDArray


class NeuralNet(nn.Module):
    def __init__(self, layers: Sequence[int]):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    @check_shapes(
        "x: [n_batch, n_inputs]",
        "return: [n_batch, n_outputs]",
    )
    def forward(self, x: tc.Tensor) -> tc.Tensor:
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        x = self.layers[-1](x)
        return x


def make_torch_nn_problem(
    name: str,
    model: nn.Module,
    loss_fn: Callable[[], tc.Tensor],
    min_param: float = -100.0,
    max_param: float = 100.0,
) -> Problem:
    n_param = sum(tc.numel(parameter) for parameter in model.parameters())

    @problem(
        domain_lower=n_param * [min_param],
        domain_upper=n_param * [max_param],
        known_optima=[],
        name=name,
    )
    def torch_nn_problem(x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
        flat = Flattener(tuple(x.shape[:-1]))
        flat_x = flat.flatten(x)
        flat_losses = np.zeros_like(flat_x[:, 0])
        flat_grads = np.zeros_like(flat_x)

        for i in range(flat.size):
            with tc.no_grad():
                j = 0
                for parameter in model.parameters():
                    k = j + tc.numel(parameter)
                    xp = np.reshape(flat_x[i, j:k], parameter.shape)
                    parameter[...] = tc.tensor(xp, dtype=parameter.dtype)
                    j = k
            model.zero_grad()
            loss = loss_fn()
            loss.backward()  # type: ignore[no-untyped-call]
            with tc.no_grad():
                flat_losses[i] = loss.item()
                j = 0
                for parameter in model.parameters():
                    k = j + tc.numel(parameter)
                    g = parameter.grad
                    assert g is not None
                    flat_grads[i, j:k] = np.reshape(g.numpy(), [-1])
                    j = k

        return flat.unflatten(flat_losses), flat.unflatten(flat_grads)

    return torch_nn_problem


@check_shapes(
    "x: [n_rows, n_inputs]",
    "y: [n_rows, n_outputs]",
)
def make_nn_model(name: str, x: tc.Tensor, y: tc.Tensor, hidden: Sequence[int] = (100,)) -> Problem:
    n_inputs = x.shape[1]
    n_outputs = y.shape[1]
    model = NeuralNet([n_inputs] + list(hidden) + [n_outputs])
    return make_torch_nn_problem(name, model, lambda: F.mse_loss(model(x), y))


make_nn_model(
    "sin_nn", tc.linspace(0.0, 10.0, 25)[:, None], tc.sin(tc.linspace(0.0, 10.0, 25)[:, None])
)

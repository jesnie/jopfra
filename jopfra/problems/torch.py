from abc import abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import Callable, TypeAlias

import numpy as np
import torch as tc
from check_shapes import check_shapes, inherit_check_shapes
from torch import nn
from torch.nn import functional as F

from jopfra.api import Problem
from jopfra.flatten import Flattener
from jopfra.paths import MatplotlibPngFile, MiscDir
from jopfra.problems.api import problem
from jopfra.problems.datasets import (
    DatasetFactory,
    baby_size_dataset,
    fetal_size_dataset,
    get_dataset,
    gpflow_example_1_dataset,
    gpflow_example_2_dataset,
    gpflow_example_3_dataset,
    gpflow_example_4_dataset,
    gpflow_example_5_dataset,
    gpflow_example_6_dataset,
    mauna_loa_co2_dataset,
    sin_dataset,
)
from jopfra.types import AnyNDArray


class ProblemModule(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    @check_shapes(
        "return: []",
    )
    def loss(self) -> tc.Tensor: ...

    def plot(self, dest: MiscDir) -> None:
        pass


ProblemModuleFactory: TypeAlias = Callable[[], ProblemModule]


@lru_cache(maxsize=1)
def get_loss_module(factory: ProblemModuleFactory) -> ProblemModule:
    return factory()


def make_torch_module_problem(
    module_factory: ProblemModuleFactory,
    *,
    name: str | None = None,
    min_param: float = -100.0,
    max_param: float = 100.0,
) -> Problem:
    module = get_loss_module(module_factory)
    if name is None:
        name = module.name
    assert name is not None
    n_param = sum(tc.numel(parameter) for parameter in module.parameters())
    del module

    @check_shapes(
        "x: [n_inputs]",
    )
    def plot_torch_module_problem(dest: MiscDir, x: AnyNDArray) -> None:
        module = get_loss_module(module_factory)
        with tc.no_grad():
            i = 0
            for parameter in module.parameters():
                j = i + tc.numel(parameter)
                xp = np.reshape(x[i:j], parameter.shape)
                parameter[...] = tc.tensor(xp, dtype=parameter.dtype)
                i = j
        module.plot(dest)

    @problem(
        domain_lower=n_param * [min_param],
        domain_upper=n_param * [max_param],
        known_optima=[],
        name=name,
        plot=plot_torch_module_problem,
    )
    def torch_module_problem(x: AnyNDArray) -> tuple[AnyNDArray, AnyNDArray]:
        module = get_loss_module(module_factory)

        flat = Flattener(tuple(x.shape[:-1]))
        flat_x = flat.flatten(x)
        flat_losses = np.zeros_like(flat_x[:, 0])
        flat_grads = np.zeros_like(flat_x)

        for i in range(flat.size):
            with tc.no_grad():
                j = 0
                for parameter in module.parameters():
                    k = j + tc.numel(parameter)
                    xp = np.reshape(flat_x[i, j:k], parameter.shape)
                    parameter[...] = tc.tensor(xp, dtype=parameter.dtype)
                    j = k
            module.zero_grad()
            loss = module.loss()
            loss.backward()  # type: ignore[no-untyped-call]
            with tc.no_grad():
                flat_losses[i] = loss.item()
                j = 0
                for parameter in module.parameters():
                    k = j + tc.numel(parameter)
                    g = parameter.grad
                    assert g is not None
                    flat_grads[i, j:k] = np.reshape(g.numpy(), [-1])
                    j = k

        return flat.unflatten(flat_losses), flat.unflatten(flat_grads)

    return torch_module_problem


class NeuralNet(ProblemModule):
    def __init__(self, dataset_factory: DatasetFactory, hidden: Sequence[int]):
        ds = get_dataset(dataset_factory)
        super().__init__(f"{ds.name}_nn_{'_'.join(str(i) for i in hidden)}")

        layers = [ds.n_inputs] + list(hidden) + [ds.n_outputs]

        self.dataset_factory = dataset_factory
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

    @inherit_check_shapes
    def loss(self) -> tc.Tensor:
        ds = get_dataset(self.dataset_factory)
        x = ds.x.float()
        y = ds.y.float()
        return F.mse_loss(self(x), y)

    def plot(self, dest: MiscDir) -> None:
        ds = get_dataset(self.dataset_factory)

        if ds.n_inputs != 1:
            return

        x_min = tc.min(ds.x).item()
        x_max = tc.max(ds.x).item()
        x_delta = x_max - x_min
        x_plot = tc.linspace(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta, 200)[:, None]

        y_min = tc.min(ds.y).item()
        y_max = tc.max(ds.y).item()
        y_delta = y_max - y_min
        y_min -= 0.1 * y_delta
        y_max += 0.1 * y_delta
        y_plot = self(x_plot).detach()

        png = dest.get("pred.png", MatplotlibPngFile)
        with png.subplots(1, 2, figsize=(20, 12)) as (_, axs):
            ax1, ax2 = axs

            ax1.set_title(f"L: {self.loss().item():.4}")
            ax1.plot(x_plot, y_plot)
            ax1.scatter(ds.x, ds.y)
            ax1.set_ylim(y_min, y_max)

            ax2.set_title(f"L: {self.loss().item():.4}")
            ax2.plot(x_plot, y_plot)
            ax2.scatter(ds.x, ds.y)

    @staticmethod
    def factory(
        dataset_factory: DatasetFactory, hidden: Sequence[int] = (100,)
    ) -> ProblemModuleFactory:
        return lambda: NeuralNet(dataset_factory, hidden)


make_torch_module_problem(NeuralNet.factory(sin_dataset))
make_torch_module_problem(NeuralNet.factory(fetal_size_dataset))
make_torch_module_problem(NeuralNet.factory(baby_size_dataset))
make_torch_module_problem(NeuralNet.factory(mauna_loa_co2_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_1_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_2_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_3_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_4_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_5_dataset))
make_torch_module_problem(NeuralNet.factory(gpflow_example_6_dataset))

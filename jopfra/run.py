import sys
from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from jopfra.minimisers.api import iter_minimisers
from jopfra.paths import ProblemMinimiserResultDir, ProblemResultDir, setup_dest
from jopfra.problems.api import problems
from jopfra.problems.monitoring import LoggingProblem


def log_problem_minimiser_results(
    pm_dest: ProblemMinimiserResultDir, pm_results: pd.DataFrame
) -> None:
    pm_dest.result.write(pm_results)


def log_problem_results(p_dest: ProblemResultDir, p_results: Mapping[str, pd.DataFrame]) -> None:
    loss_plots: tuple[tuple[str, str, Callable[[pd.DataFrame], "pd.Series[float]"]], ...] = (
        (
            "Loss vs problem calls.",
            "Number of calls to the problem.",
            lambda df: df.n_calls,
        ),
        (
            "Loss vs problem evaluations.",
            "Number of evaluations of the problem.",
            lambda df: df.n_evals,
        ),
        (
            "Loss vs time.",
            "Time / seconds.",
            lambda df: df.time_ns / (10**9),
        ),
    )

    def setup_raw(ax: Axes) -> None:
        pass

    def setup_truncated(ax: Axes) -> None:
        data = np.concatenate([l.get_ydata() for l in ax.lines])  # type: ignore[attr-defined]
        lower = np.min(data)
        data = data[data != lower]
        upper = np.minimum(np.max(data), 10 * np.quantile(data, 0.1))
        ax.set_ylim(lower, upper)

    def setup_log(ax: Axes) -> None:
        ax.set_yscale("log")

    setup_axs = {
        "": setup_raw,
        " (zoom)": setup_truncated,
        " (log)": setup_log,
    }

    with p_dest.plots.subplots(len(loss_plots), len(setup_axs), figsize=(20.0, 25.0)) as (fig, axs):
        for loss_plot, loss_axs in zip(loss_plots, axs):
            title, x_label, x_fn = loss_plot
            for (title_suffix, setup_ax), loss_ax in zip(setup_axs.items(), loss_axs):
                for m_name, pm_results in p_results.items():
                    loss_ax.plot(
                        x_fn(pm_results),
                        pm_results.loss,
                        label=m_name,
                    )

                setup_ax(loss_ax)
                loss_ax.set_title(title + title_suffix)
                loss_ax.set_xlabel(x_label)
                loss_ax.set_ylabel("Loss")
                loss_ax.legend()

        fig.tight_layout()


def main() -> None:
    parser = ArgumentParser(description="Run optimisation experiments.")
    parser.add_argument(
        "--problems",
        "-p",
        default=[],
        type=str,
        nargs="+",
        help=f"Problems to optimise: Choose from {sorted(problems)}",
    )
    parser.add_argument(
        "--minimisers",
        "-m",
        default=[],
        type=str,
        nargs="+",
        help=f"Minimisers for optimisation: Choose from {sorted(iter_minimisers)}",
    )
    parser.add_argument(
        "--n_iter",
        "--n-iter",
        "-n",
        default=100,
        type=int,
        help="Number of iterations to run.",
    )
    parser.add_argument(
        "--dest",
        "-d",
        default=Path(__file__).parent.parent / "results",
        type=Path,
        help="Where to write output.",
    )

    args = parser.parse_args()
    assert args.problems, (args.problems, sorted(problems))
    assert all(p in problems for p in args.problems), (args.problems, sorted(problems))
    assert args.minimisers, (args.minimisers, sorted(iter_minimisers))
    assert all(p in iter_minimisers for p in args.minimisers), (
        args.minimisers,
        sorted(iter_minimisers),
    )

    dest = setup_dest(args.dest)
    dest.metadata.write(
        {
            "argv": sys.argv,
        },
        indent=2,
    )

    for p_name in args.problems:
        p_results: dict[str, pd.DataFrame] = {}
        p_dest = dest.problems[p_name]

        for m_name in args.minimisers:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(p_name)
            print(m_name)
            problem = LoggingProblem(problems[p_name])
            minimiser = iter_minimisers[m_name]

            n_calls = []
            n_evals = []
            times_ns = []
            xs = []
            losses = []
            for _, y in zip(range(args.n_iter), minimiser.iter_minimise(problem, ())):
                n_calls.append(problem.n_calls)
                n_evals.append(problem.n_evals)
                times_ns.append(problem.time_ns)
                xs.append(y.x.tolist())
                losses.append(float(y.loss))

            pm_results = pd.DataFrame(
                {
                    "n_calls": n_calls,
                    "n_evals": n_evals,
                    "time_ns": times_ns,
                    "xs": xs,
                    "loss": losses,
                }
            )
            pm_dest = p_dest.minimisers[m_name]

            p_results[m_name] = pm_results
            log_problem_minimiser_results(pm_dest, pm_results)
            problem.plot(pm_dest.plots, xs[-1])

        log_problem_results(p_dest, p_results)

    print(f"Output written to: {dest}")


if __name__ == "__main__":
    main()

import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import typedpath as tp
import typedpath.inspect as tpi
from git.exc import InvalidGitRepositoryError
from git.repo import Repo


class MatplotlibPngFile(tp.TypedFile):
    default_suffix = ".png"

    @contextmanager
    def subplots(self, *args: Any, **kwargs: Any) -> Iterator[tuple[Any, Any]]:
        fig, axs = plt.subplots(*args, **kwargs)
        yield (fig, axs)
        fig.savefig(self.write_path())
        plt.close(fig)


class MiscDir(tp.TypedDir):
    default_suffix = ""

    def get(self, filename: str, t: type[tpi.TP], **kwargs: Any) -> tpi.TP:
        args = tp.withargs(**kwargs)
        return tpi.make(t, self.pretty_path() / filename, args)


class ProblemMinimiserResultDir(tp.StructDir):
    result: tp.PandasCsvFile
    plots: MiscDir


class ProblemResultDir(tp.StructDir):
    minimisers: tp.DictDir[str, ProblemMinimiserResultDir]
    plots: MatplotlibPngFile


class ResultDir(tp.StructDir):
    metadata: tp.JSONFile
    problems: tp.DictDir[str, ProblemResultDir]


def setup_dest(root: Path) -> ResultDir:
    root.mkdir(parents=True, exist_ok=True)

    script_name = Path(sys.argv[0]).stem
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    branch_name = "NO_BRANCH"
    with suppress(InvalidGitRepositoryError):  # Keep current/default branch_name
        repo = Repo(__file__, search_parent_directories=True)
        with suppress(TypeError):  # Keep current/default branch_name
            branch_name = str(repo.active_branch)

    run_id = f"{script_name}_{branch_name}_{timestamp}".replace("/", "_")

    dest = root / run_id
    dest.mkdir()

    latest_dir = root / "latest"
    if latest_dir.is_symlink():
        latest_dir.unlink()
    latest_dir.symlink_to(run_id)

    return ResultDir(dest)

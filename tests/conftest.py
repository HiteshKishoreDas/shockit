from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--plot-tests",
        action="store_true",
        default=False,
        help="Save PNG plot artifacts for tests that support plotting.",
    )
    parser.addoption(
        "--plot-dir",
        action="store",
        default="test_artifacts/plots",
        help="Directory where test plot artifacts are written.",
    )


@dataclass
class TestPlotter:
    enabled: bool
    output_dir: Path

    def save_slice(
        self,
        name: str,
        array2d,
        *,
        title: str | None = None,
        cmap: str = "viridis",
        colorbar_label: str | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{name}.png"

        figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = axis.imshow(np.asarray(array2d), origin="lower", cmap=cmap, aspect="auto")
        axis.set_xlabel("j")
        axis.set_ylabel("i")
        axis.set_title(title or name.replace("_", " "))
        colorbar = figure.colorbar(image, ax=axis)
        if colorbar_label:
            colorbar.set_label(colorbar_label)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    def save_line(
        self,
        name: str,
        x,
        y,
        *,
        title: str | None = None,
        xlabel: str = "x",
        ylabel: str = "y",
        label: str | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{name}.png"

        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot(np.asarray(x), np.asarray(y), label=label)
        axis.set_title(title or name.replace("_", " "))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if label:
            axis.legend()
        axis.grid(alpha=0.3)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path


@pytest.fixture
def test_plotter(pytestconfig: pytest.Config, request: pytest.FixtureRequest) -> TestPlotter:
    raw_name = request.node.nodeid.replace("/", "__").replace("::", "__")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)
    output_dir = Path(pytestconfig.getoption("--plot-dir")) / safe_name
    return TestPlotter(
        enabled=bool(pytestconfig.getoption("--plot-tests")),
        output_dir=output_dir,
    )

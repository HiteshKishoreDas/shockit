"""Legacy `.npz` pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import NpzChunkedInput, NpzChunkedOutput, join_chunked_field, run_chunked_shock_finder
from .config import ShockFinderConfig
from .fields import FluidCube
from .finder import ShockFinder


@dataclass(frozen=True)
class NpzLayout:
    """Resolved input layout for a simulation data directory."""

    kind: str
    input_path: Path


def detect_npz_layout(root: str | Path) -> NpzLayout:
    """Detect whether a directory contains plain or chunked shock-finder inputs."""

    root_path = Path(root)
    chunked_root = root_path / "chunked_npz"
    if _has_chunked_inputs(chunked_root):
        return NpzLayout(kind="chunked", input_path=chunked_root)
    if _has_plain_inputs(root_path):
        return NpzLayout(kind="plain", input_path=root_path)
    if _has_chunked_inputs(root_path):
        return NpzLayout(kind="chunked", input_path=root_path)
    raise FileNotFoundError(
        f"Could not find plain .npz inputs or a chunked_npz layout under {root_path}."
    )


def run_npz_shock_finder(
    root: str | Path,
    config: ShockFinderConfig,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    gamma: float = 5.0 / 3.0,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the shock finder on either plain or chunked `.npz` inputs."""

    layout = detect_npz_layout(root)
    if layout.kind == "chunked":
        input_store = NpzChunkedInput(layout.input_path)
        output_store = NpzChunkedOutput(layout.input_path.parent / "shock_chunked_npz", input_store.layout)
        return run_chunked_shock_finder(
            input_store,
            output_store,
            config=config,
            dx=dx,
            dy=dy,
            dz=dz,
            gamma=gamma,
            progress=progress,
        )

    cube = _load_plain_cube(
        layout.input_path,
        dx=dx,
        dy=dy,
        dz=dz,
        gamma=gamma,
    )
    result = ShockFinder(config).find(cube, progress=progress)
    _save_plain_outputs(layout.input_path, result)
    return result.summary


def _has_plain_inputs(root: Path) -> bool:
    return all((root / filename).exists() for filename in ("rho.npz", "prs.npz", "v1.npz", "v2.npz", "v3.npz"))


def _has_chunked_inputs(root: Path) -> bool:
    return all((root / field).is_dir() for field in ("rho", "prs", "v1", "v2", "v3"))


def _load_plain_cube(
    root: Path,
    *,
    dx: float,
    dy: float,
    dz: float,
    gamma: float,
) -> FluidCube:
    return FluidCube(
        rho=np.load(root / "rho.npz")["arr_0"],
        pressure=np.load(root / "prs.npz")["arr_0"],
        vx=np.load(root / "v1.npz")["arr_0"],
        vy=np.load(root / "v2.npz")["arr_0"],
        vz=np.load(root / "v3.npz")["arr_0"],
        dx=dx,
        dy=dy,
        dz=dz,
        gamma=gamma,
    )


def _save_plain_outputs(root: Path, result) -> None:
    np.savez(root / "mask.npz", result.shock_mask)
    np.savez(root / "mask_full.npz", result.full_shock_mask)
    np.savez(root / "mach_T.npz", result.mach_temperature)
    np.savez(root / "mach_prs.npz", result.mach_pressure)


__all__ = ["NpzLayout", "detect_npz_layout", "run_npz_shock_finder", "join_chunked_field"]

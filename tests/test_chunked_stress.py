from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from shockit import FluidCube, ShockFinder, ShockFinderConfig
from shockit.chunking.layout import iter_balanced_slices
from shockit.result import ChunkedShockFinderResult

pytestmark = pytest.mark.stress

_DEFAULT_STRESS_SHAPE = (256, 256, 256)
_DEFAULT_STRESS_CHUNK_SHAPE = (128, 128, 128)
_DEFAULT_STRESS_SEED = 20260628


def _parse_shape_env(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    parts = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise ValueError(f"{name} must be set to one integer or three comma-separated integers.")


def _stress_shape() -> tuple[int, int, int]:
    return _parse_shape_env("SHOCKIT_STRESS_SHAPE", _DEFAULT_STRESS_SHAPE)


def _stress_chunk_shape() -> tuple[int, int, int]:
    return _parse_shape_env("SHOCKIT_STRESS_CHUNK_SHAPE", _DEFAULT_STRESS_CHUNK_SHAPE)


def _stress_enabled() -> bool:
    return os.environ.get("SHOCKIT_RUN_STRESS_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _chunk_filename(chunk_index: int, chunk_slice: tuple[slice, slice, slice]) -> str:
    return (
        f"chunk_{chunk_index:04d}"
        f"_x{chunk_slice[0].start:04d}-{chunk_slice[0].stop:04d}"
        f"_y{chunk_slice[1].start:04d}-{chunk_slice[1].stop:04d}"
        f"_z{chunk_slice[2].start:04d}-{chunk_slice[2].stop:04d}.npz"
    )


def _write_chunked_cube(
    root: Path,
    *,
    shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    seed: int,
) -> None:
    for field_name in ("rho", "prs", "v1", "v2", "v3"):
        (root / field_name).mkdir(parents=True, exist_ok=True)

    # Generate each chunk independently so the stress path can scale up to
    # SHOCKIT_STRESS_SHAPE=2048 without materializing the whole cube in memory.
    for chunk_index, chunk_slice in enumerate(iter_balanced_slices(shape, chunk_shape)):
        fields = _make_synthetic_chunk(shape=shape, chunk_slice=chunk_slice, seed=seed)
        start = np.array([chunk_slice[0].start, chunk_slice[1].start, chunk_slice[2].start], dtype=np.int64)
        stop = np.array([chunk_slice[0].stop, chunk_slice[1].stop, chunk_slice[2].stop], dtype=np.int64)
        for field_name, values in fields.items():
            np.savez(
                root / field_name / _chunk_filename(chunk_index, chunk_slice),
                values=values,
                start=start,
                stop=stop,
                full_shape=np.array(shape, dtype=np.int64),
            )


def _make_synthetic_chunk(
    *,
    shape: tuple[int, int, int],
    chunk_slice: tuple[slice, slice, slice],
    seed: int,
) -> dict[str, np.ndarray]:
    x = (np.arange(chunk_slice[0].start, chunk_slice[0].stop, dtype=np.float32) + 0.5) / shape[0]
    y = (np.arange(chunk_slice[1].start, chunk_slice[1].stop, dtype=np.float32) + 0.5) / shape[1]
    z = (np.arange(chunk_slice[2].start, chunk_slice[2].stop, dtype=np.float32) + 0.5) / shape[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    phase = np.float32((seed % 997) / 997.0 * 2.0 * math.pi)
    coarse = _smooth_noise(xx, yy, zz, phase=phase)
    fine = _smooth_noise(xx, yy, zz, phase=phase + np.float32(1.234), scale=np.float32(2.7))

    front = xx + np.float32(0.035) * coarse > np.float32(0.52)
    upstream_rho = np.float32(1.0) * (np.float32(1.0) + np.float32(0.04) * fine)
    downstream_rho = np.float32(2.6) * (np.float32(1.0) + np.float32(0.04) * fine)
    upstream_pressure = np.float32(1.0) * (np.float32(1.0) + np.float32(0.05) * coarse)
    downstream_pressure = np.float32(7.0) * (np.float32(1.0) + np.float32(0.05) * coarse)

    vx_upstream = np.float32(2.2) + np.float32(0.08) * coarse
    vx_downstream = np.float32(0.85) + np.float32(0.06) * fine
    vy = np.float32(0.08) * _smooth_noise(xx, yy, zz, phase=phase + np.float32(2.468), scale=np.float32(1.8))
    vz = np.float32(0.08) * _smooth_noise(xx, yy, zz, phase=phase + np.float32(3.579), scale=np.float32(2.2))

    return {
        "rho": np.where(front, downstream_rho, upstream_rho).astype(np.float32, copy=False),
        "prs": np.where(front, downstream_pressure, upstream_pressure).astype(np.float32, copy=False),
        "v1": np.where(front, vx_downstream, vx_upstream).astype(np.float32, copy=False),
        "v2": vy.astype(np.float32, copy=False),
        "v3": vz.astype(np.float32, copy=False),
    }


def _smooth_noise(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    *,
    phase: np.float32,
    scale: np.float32 = np.float32(1.0),
) -> np.ndarray:
    two_pi = np.float32(2.0 * math.pi)
    signal = (
        np.sin(two_pi * scale * (np.float32(1.0) * xx + np.float32(0.7) * yy + np.float32(0.4) * zz) + phase)
        + np.float32(0.6)
        * np.cos(two_pi * scale * (np.float32(0.3) * xx + np.float32(1.1) * yy + np.float32(0.8) * zz) + np.float32(1.7) * phase)
        + np.float32(0.35)
        * np.sin(two_pi * scale * (np.float32(1.6) * xx + np.float32(0.2) * yy + np.float32(1.3) * zz) + np.float32(0.5) * phase)
    )
    return signal.astype(np.float32, copy=False)


def _chunked_cube(root: Path) -> FluidCube:
    return FluidCube(
        rho=root / "rho",
        pressure=root / "prs",
        vx=root / "v1",
        vy=root / "v2",
        vz=root / "v3",
    )


@pytest.mark.stress
def test_chunked_pipeline_stress_large_synthetic_cube(tmp_path: Path) -> None:
    if not _stress_enabled():
        pytest.skip("set SHOCKIT_RUN_STRESS_TESTS=1 to opt into the chunked stress test")

    shape = _stress_shape()
    chunk_shape = _stress_chunk_shape()
    seed = int(os.environ.get("SHOCKIT_STRESS_SEED", str(_DEFAULT_STRESS_SEED)))

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_chunked_cube(
        input_root,
        shape=shape,
        chunk_shape=chunk_shape,
        seed=seed,
    )

    result = ShockFinder(
        ShockFinderConfig(
            reduce_to_centers=False,
            min_mach=1.1,
            sampling_method="nearest_axis",
        )
    ).find(
        _chunked_cube(input_root),
        output=output_root,
        progress=False,
    )

    assert isinstance(result, ChunkedShockFinderResult)
    assert result.summary["grid_shape"] == shape
    assert result.summary["total_cells"] == math.prod(shape)
    assert result.shock_mask.shape == shape
    assert result.full_shock_mask is not None
    assert result.summary["full_shock_cells"] >= result.summary["shock_cells"]
    assert (output_root / "summary.json").exists()

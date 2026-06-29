from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shockit import FluidCube, ShockFinder, ShockFinderConfig
from shockit.chunking import save_chunked_field
from shockit.result import ChunkedShockFinderResult, ShockFinderResult


def _in_memory_cube() -> FluidCube:
    return FluidCube(
        rho=np.ones((4, 4, 4)),
        pressure=np.ones((4, 4, 4)),
        vx=np.zeros((4, 4, 4)),
        vy=np.zeros((4, 4, 4)),
        vz=np.zeros((4, 4, 4)),
    )


def _chunked_cube(root: Path) -> FluidCube:
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)
    return FluidCube(
        rho=root / "rho",
        pressure=root / "prs",
        vx=root / "v1",
        vy=root / "v2",
        vz=root / "v3",
    )


def test_find_returns_in_memory_result_for_array_backed_cube() -> None:
    result = ShockFinder(ShockFinderConfig(min_mach=1.1, reduce_to_centers=False)).find(
        _in_memory_cube(),
        progress=False,
    )

    assert isinstance(result, ShockFinderResult)


def test_find_returns_chunked_result_for_path_backed_cube(tmp_path: Path) -> None:
    result = ShockFinder(ShockFinderConfig(min_mach=1.1, reduce_to_centers=False)).find(
        _chunked_cube(tmp_path / "input"),
        output=tmp_path / "output",
        progress=False,
    )

    assert isinstance(result, ChunkedShockFinderResult)


def test_find_keeps_chunked_outputs_unreduced_even_when_center_reduction_is_requested(tmp_path: Path) -> None:
    result = ShockFinder(ShockFinderConfig(min_mach=1.1, reduce_to_centers=True)).find(
        _chunked_cube(tmp_path / "input"),
        output=tmp_path / "output",
        progress=False,
    )

    assert isinstance(result, ChunkedShockFinderResult)
    np.testing.assert_array_equal(result.shock_mask.load(), result.full_shock_mask.load())
    assert result.summary["reduce_to_centers"] is False
    assert result.summary["requested_reduce_to_centers"] is True


def test_find_rejects_output_for_in_memory_cube(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="save_result_hdf5"):
        ShockFinder(ShockFinderConfig()).find(
            _in_memory_cube(),
            output=tmp_path / "output",
            progress=False,
        )


def test_find_rejects_non_fluid_cube_input() -> None:
    with pytest.raises(TypeError, match="FluidCube"):
        ShockFinder(ShockFinderConfig()).find(np.ones((4, 4, 4)), progress=False)  # type: ignore[arg-type]

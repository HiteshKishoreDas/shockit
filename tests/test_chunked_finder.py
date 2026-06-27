from __future__ import annotations

import json

import numpy as np

from shockit import (
    ShockFinder,
    ShockFinderConfig,
    join_chunked_field,
    run_chunked_npz_shock_finder,
    save_chunked_field,
)
from tests.helpers import make_planar_shock_cube


def test_chunked_npz_shock_finder_matches_full_cube(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    save_chunked_field("rho", cube.rho, input_root, target_chunk_size=7)
    save_chunked_field("prs", cube.pressure, input_root, target_chunk_size=7)
    save_chunked_field("v1", cube.vx, input_root, target_chunk_size=7)
    save_chunked_field("v2", cube.vy, input_root, target_chunk_size=7)
    save_chunked_field("v3", cube.vz, input_root, target_chunk_size=7)

    config = ShockFinderConfig(
        chunk_size=None,
        reduce_to_centers=False,
        min_mach=1.1,
        sampling_method="nearest_axis",
    )
    summary = run_chunked_npz_shock_finder(
        input_root,
        output_root,
        config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(
        join_chunked_field(output_root / "shock_mask"),
        full_result.shock_mask,
    )
    np.testing.assert_array_equal(
        join_chunked_field(output_root / "shock_zone_mask"),
        full_result.shock_zone_mask,
    )
    np.testing.assert_array_equal(
        join_chunked_field(output_root / "full_shock_mask"),
        full_result.full_shock_mask,
    )
    np.testing.assert_allclose(
        join_chunked_field(output_root / "mach_pressure"),
        full_result.mach_pressure,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        join_chunked_field(output_root / "mach_temperature"),
        full_result.mach_temperature,
        equal_nan=True,
    )
    assert summary["shock_cells"] == full_result.summary["shock_cells"]
    assert summary["full_shock_cells"] == full_result.summary["full_shock_cells"]
    stored_summary = json.loads((output_root / "summary.json").read_text())
    assert stored_summary["shock_cells"] == full_result.summary["shock_cells"]


def test_chunked_npz_shock_finder_center_reduction_matches_full_cube(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"

    save_chunked_field("rho", cube.rho, input_root, target_chunk_size=7)
    save_chunked_field("prs", cube.pressure, input_root, target_chunk_size=7)
    save_chunked_field("v1", cube.vx, input_root, target_chunk_size=7)
    save_chunked_field("v2", cube.vy, input_root, target_chunk_size=7)
    save_chunked_field("v3", cube.vz, input_root, target_chunk_size=7)

    config = ShockFinderConfig(
        chunk_size=None,
        reduce_to_centers=True,
        min_mach=1.1,
        sampling_method="nearest_axis",
    )
    summary = run_chunked_npz_shock_finder(
        input_root,
        output_root,
        config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(
        join_chunked_field(output_root / "shock_mask"),
        full_result.shock_mask,
    )
    assert summary["shock_cells"] == full_result.summary["shock_cells"]
    assert summary["n_connected_components"] == full_result.summary["n_connected_components"]

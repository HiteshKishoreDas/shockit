from __future__ import annotations

import json

import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from shockit.chunking import NpzChunkedInput, NpzChunkedOutput, join_chunked_field, run_chunked_shock_finder, save_chunked_field
from tests.helpers import make_oblique_shock_cube, make_periodic_edge_shock_cube, make_planar_shock_cube


def _write_chunked_input(cube, root, chunk_size: int) -> None:
    save_chunked_field("rho", cube.rho, root, target_chunk_size=chunk_size)
    save_chunked_field("prs", cube.pressure, root, target_chunk_size=chunk_size)
    save_chunked_field("v1", cube.vx, root, target_chunk_size=chunk_size)
    save_chunked_field("v2", cube.vy, root, target_chunk_size=chunk_size)
    save_chunked_field("v3", cube.vz, root, target_chunk_size=chunk_size)


def test_chunked_runner_planar_matches_full_cube(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_chunked_input(cube, input_root, chunk_size=7)

    config = ShockFinderConfig(
        reduce_to_centers=False,
        min_mach=1.1,
        sampling_method="nearest_axis",
    )
    input_store = NpzChunkedInput(input_root)
    output_store = NpzChunkedOutput(output_root, input_store.layout)
    summary = run_chunked_shock_finder(
        input_store,
        output_store,
        config=config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_mask"), full_result.shock_mask)
    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_zone_mask"), full_result.shock_zone_mask)
    np.testing.assert_array_equal(join_chunked_field(output_root / "full_shock_mask"), full_result.full_shock_mask)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_pressure"), full_result.mach_pressure, equal_nan=True)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_temperature"), full_result.mach_temperature, equal_nan=True)
    assert summary["shock_cells"] == full_result.summary["shock_cells"]
    assert summary["full_shock_cells"] == full_result.summary["full_shock_cells"]
    stored_summary = json.loads((output_root / "summary.json").read_text())
    assert stored_summary["shock_cells"] == full_result.summary["shock_cells"]


def test_chunked_runner_oblique_matches_full_cube(tmp_path) -> None:
    cube = make_oblique_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_chunked_input(cube, input_root, chunk_size=7)

    config = ShockFinderConfig(
        reduce_to_centers=False,
        min_mach=1.1,
        sampling_method="trilinear",
    )
    input_store = NpzChunkedInput(input_root)
    output_store = NpzChunkedOutput(output_root, input_store.layout)
    run_chunked_shock_finder(
        input_store,
        output_store,
        config=config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_zone_mask"), full_result.shock_zone_mask)
    np.testing.assert_array_equal(join_chunked_field(output_root / "full_shock_mask"), full_result.full_shock_mask)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_pressure"), full_result.mach_pressure, equal_nan=True, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_temperature"), full_result.mach_temperature, equal_nan=True, atol=1e-10, rtol=1e-10)


def test_chunked_runner_center_reduction_matches_full_cube(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_chunked_input(cube, input_root, chunk_size=7)

    config = ShockFinderConfig(
        reduce_to_centers=True,
        min_mach=1.1,
        sampling_method="nearest_axis",
    )
    input_store = NpzChunkedInput(input_root)
    output_store = NpzChunkedOutput(output_root, input_store.layout)
    summary = run_chunked_shock_finder(
        input_store,
        output_store,
        config=config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_mask"), full_result.shock_mask)
    assert summary["shock_cells"] == full_result.summary["shock_cells"]
    assert summary["n_connected_components"] == full_result.summary["n_connected_components"]


def test_chunked_runner_periodic_edge_matches_full_cube_without_center_reduction(tmp_path) -> None:
    cube = make_periodic_edge_shock_cube(mach=2.0, n=18)
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_chunked_input(cube, input_root, chunk_size=6)

    config = ShockFinderConfig(
        reduce_to_centers=False,
        min_mach=1.1,
        sampling_method="nearest_axis",
    )
    input_store = NpzChunkedInput(input_root)
    output_store = NpzChunkedOutput(output_root, input_store.layout)
    run_chunked_shock_finder(
        input_store,
        output_store,
        config=config,
        progress=False,
    )
    full_result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_zone_mask"), full_result.shock_zone_mask)
    np.testing.assert_array_equal(join_chunked_field(output_root / "full_shock_mask"), full_result.full_shock_mask)
    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_mask"), full_result.shock_mask)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_pressure"), full_result.mach_pressure, equal_nan=True)
    np.testing.assert_allclose(join_chunked_field(output_root / "mach_temperature"), full_result.mach_temperature, equal_nan=True)

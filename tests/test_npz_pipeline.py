from __future__ import annotations

import json

import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from shockit.chunking import join_chunked_field, save_chunked_field
from shockit.npz_pipeline import detect_npz_layout, run_npz_shock_finder
from tests.helpers import make_planar_shock_cube


def test_detect_npz_layout_finds_plain_inputs(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=8)
    np.savez(tmp_path / "rho.npz", cube.rho)
    np.savez(tmp_path / "prs.npz", cube.pressure)
    np.savez(tmp_path / "v1.npz", cube.vx)
    np.savez(tmp_path / "v2.npz", cube.vy)
    np.savez(tmp_path / "v3.npz", cube.vz)

    layout = detect_npz_layout(tmp_path)
    assert layout.kind == "plain"
    assert layout.input_path == tmp_path


def test_detect_npz_layout_prefers_chunked_subdirectory(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=8)
    chunk_root = tmp_path / "chunked_npz"
    save_chunked_field("rho", cube.rho, chunk_root, target_chunk_size=4)
    save_chunked_field("prs", cube.pressure, chunk_root, target_chunk_size=4)
    save_chunked_field("v1", cube.vx, chunk_root, target_chunk_size=4)
    save_chunked_field("v2", cube.vy, chunk_root, target_chunk_size=4)
    save_chunked_field("v3", cube.vz, chunk_root, target_chunk_size=4)

    layout = detect_npz_layout(tmp_path)
    assert layout.kind == "chunked"
    assert layout.input_path == chunk_root


def test_run_npz_shock_finder_plain_matches_direct_finder(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=12)
    np.savez(tmp_path / "rho.npz", cube.rho)
    np.savez(tmp_path / "prs.npz", cube.pressure)
    np.savez(tmp_path / "v1.npz", cube.vx)
    np.savez(tmp_path / "v2.npz", cube.vy)
    np.savez(tmp_path / "v3.npz", cube.vz)

    config = ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)
    summary = run_npz_shock_finder(tmp_path, config, progress=False)
    result = ShockFinder(config).find(cube, progress=False)

    np.testing.assert_array_equal(np.load(tmp_path / "mask.npz")["arr_0"], result.shock_mask)
    np.testing.assert_array_equal(np.load(tmp_path / "mask_full.npz")["arr_0"], result.full_shock_mask)
    np.testing.assert_allclose(np.load(tmp_path / "mach_prs.npz")["arr_0"], result.mach_pressure, equal_nan=True)
    assert summary["shock_cells"] == result.summary["shock_cells"]


def test_run_npz_shock_finder_chunked_matches_direct_chunked_pipeline(tmp_path) -> None:
    cube = make_planar_shock_cube(mach=2.0, n=12)
    chunk_root = tmp_path / "chunked_npz"
    save_chunked_field("rho", cube.rho, chunk_root, target_chunk_size=5)
    save_chunked_field("prs", cube.pressure, chunk_root, target_chunk_size=5)
    save_chunked_field("v1", cube.vx, chunk_root, target_chunk_size=5)
    save_chunked_field("v2", cube.vy, chunk_root, target_chunk_size=5)
    save_chunked_field("v3", cube.vz, chunk_root, target_chunk_size=5)

    config = ShockFinderConfig(reduce_to_centers=True, min_mach=1.1)
    summary = run_npz_shock_finder(tmp_path, config, progress=False)
    result = ShockFinder(config).find(cube, progress=False)

    output_root = tmp_path / "shock_chunked_npz"
    np.testing.assert_array_equal(join_chunked_field(output_root / "shock_mask"), result.shock_mask)
    assert summary["shock_cells"] == result.summary["shock_cells"]
    stored_summary = json.loads((output_root / "summary.json").read_text())
    assert stored_summary["shock_cells"] == result.summary["shock_cells"]

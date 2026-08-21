from __future__ import annotations

import numpy as np

from shockit import ShockFinderConfig
from shockit.chunked_center_reduction import reduce_chunked_shock_outputs
from shockit.chunking import join_chunked_field, save_chunked_field
from shockit.derived import MachFields
from shockit.masks import center_score, reduce_to_centers


def test_connected_region_reduces_to_one_cell() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    div_v = np.zeros(mask.shape)
    div_v[2, 2, 2] = -5.0
    mach_fields = MachFields(
        mach_pressure=np.ones(mask.shape),
        mach_temperature=np.ones(mask.shape),
    )
    reduced, count = reduce_to_centers(mask, div_v, mach_fields, center_score_mode="compression")
    assert count == 1
    assert np.count_nonzero(reduced) == 1
    assert reduced[2, 2, 2]


def test_center_score_mach_pressure_selects_highest_mach() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 1, 2] = True
    div_v = np.zeros(component.shape)
    mach_fields = MachFields(
        mach_pressure=np.zeros(component.shape),
        mach_temperature=np.full(component.shape, np.nan),
    )
    mach_fields.mach_pressure[1, 1, 1] = 2.0
    mach_fields.mach_pressure[1, 1, 2] = 3.0
    score = center_score(component, div_v, mach_fields, mode="mach_pressure")
    assert np.argmax(score) == np.ravel_multi_index((1, 1, 2), component.shape)


def test_center_score_compression_selects_strongest_compression() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 2, 1] = True
    div_v = np.zeros(component.shape)
    div_v[1, 1, 1] = -1.0
    div_v[1, 2, 1] = -4.0
    mach_fields = MachFields(
        mach_pressure=np.full(component.shape, np.nan),
        mach_temperature=np.full(component.shape, np.nan),
    )
    score = center_score(component, div_v, mach_fields, mode="compression")
    assert np.argmax(score) == np.ravel_multi_index((1, 2, 1), component.shape)


def test_combined_score_handles_nan_mach_and_stays_inside_component() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 1, 2] = True
    div_v = np.zeros(component.shape)
    div_v[1, 1, 1] = -2.0
    div_v[1, 1, 2] = -1.0
    mach_fields = MachFields(
        mach_pressure=np.full(component.shape, np.nan),
        mach_temperature=np.full(component.shape, np.nan),
    )
    score = center_score(component, div_v, mach_fields, mode="combined")
    best = np.unravel_index(int(np.argmax(score)), score.shape)
    assert component[best]
    assert np.isfinite(score[best])
    assert np.all(score[~component] == -np.inf)


def test_chunked_center_reduction_merges_internal_chunk_boundary(tmp_path) -> None:
    shape = (8, 4, 4)
    full_shock_mask = np.zeros(shape, dtype=bool)
    full_shock_mask[3:5, 1, 1] = True
    div_v = np.zeros(shape, dtype=float)
    div_v[4, 1, 1] = -5.0
    mach_pressure = np.ones(shape, dtype=float)
    mach_temperature = np.ones(shape, dtype=float)

    _write_chunked_center_fields(tmp_path, full_shock_mask, div_v, mach_pressure, mach_temperature, chunk_size=4)

    summary = reduce_chunked_shock_outputs(
        tmp_path,
        ShockFinderConfig(reduce_to_centers=True, center_score="compression"),
        progress=False,
    )
    reduced = join_chunked_field(tmp_path / "shock_mask")

    assert summary["n_connected_components"] == 1
    assert summary["shock_cells"] == 1
    assert np.count_nonzero(reduced) == 1
    assert reduced[4, 1, 1]


def test_chunked_center_reduction_matches_full_cube_tie_break_across_chunk_boundary(tmp_path) -> None:
    shape = (8, 4, 4)
    full_shock_mask = np.zeros(shape, dtype=bool)
    full_shock_mask[3:5, 1, 1] = True
    div_v = np.zeros(shape, dtype=float)
    mach_pressure = np.ones(shape, dtype=float)
    mach_temperature = np.ones(shape, dtype=float)

    _write_chunked_center_fields(tmp_path, full_shock_mask, div_v, mach_pressure, mach_temperature, chunk_size=4)

    full_reduced, full_count = reduce_to_centers(
        full_shock_mask,
        div_v,
        MachFields(
            mach_pressure=mach_pressure,
            mach_temperature=mach_temperature,
        ),
        center_score_mode="compression",
    )
    summary = reduce_chunked_shock_outputs(
        tmp_path,
        ShockFinderConfig(reduce_to_centers=True, center_score="compression"),
        progress=False,
    )
    reduced = join_chunked_field(tmp_path / "shock_mask")

    assert full_count == 1
    assert summary["n_connected_components"] == 1
    np.testing.assert_array_equal(reduced, full_reduced)
    assert np.count_nonzero(reduced) == 1


def test_chunked_center_reduction_merges_periodic_boundary_component(tmp_path) -> None:
    shape = (8, 4, 4)
    full_shock_mask = np.zeros(shape, dtype=bool)
    full_shock_mask[0, 1, 1] = True
    full_shock_mask[-1, 1, 1] = True
    div_v = np.zeros(shape, dtype=float)
    div_v[-1, 1, 1] = -6.0
    mach_pressure = np.ones(shape, dtype=float)
    mach_temperature = np.ones(shape, dtype=float)

    _write_chunked_center_fields(tmp_path, full_shock_mask, div_v, mach_pressure, mach_temperature, chunk_size=4)

    summary = reduce_chunked_shock_outputs(
        tmp_path,
        ShockFinderConfig(reduce_to_centers=True, center_score="compression"),
        progress=False,
    )
    reduced = join_chunked_field(tmp_path / "shock_mask")

    assert summary["n_connected_components"] == 1
    assert summary["shock_cells"] == 2
    assert np.count_nonzero(reduced) == 2
    assert reduced[0, 1, 1]
    assert reduced[-1, 1, 1]


def _write_chunked_center_fields(
    root,
    full_shock_mask: np.ndarray,
    div_v: np.ndarray,
    mach_pressure: np.ndarray,
    mach_temperature: np.ndarray,
    *,
    chunk_size: int,
) -> None:
    save_chunked_field("shock_zone_mask", full_shock_mask, root, target_chunk_size=chunk_size)
    save_chunked_field("full_shock_mask", full_shock_mask, root, target_chunk_size=chunk_size)
    save_chunked_field("shock_mask", full_shock_mask, root, target_chunk_size=chunk_size)
    save_chunked_field("div_v", div_v, root, target_chunk_size=chunk_size)
    save_chunked_field("mach_pressure", mach_pressure, root, target_chunk_size=chunk_size)
    save_chunked_field("mach_temperature", mach_temperature, root, target_chunk_size=chunk_size)

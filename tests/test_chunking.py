from __future__ import annotations

import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from shockit.chunked import iter_balanced_slices
from tests.helpers import make_oblique_shock_cube, make_planar_shock_cube


def _assert_same_result(chunked, unchunked) -> None:
    np.testing.assert_array_equal(chunked.shock_mask, unchunked.shock_mask)
    np.testing.assert_array_equal(chunked.shock_zone_mask, unchunked.shock_zone_mask)
    np.testing.assert_array_equal(chunked.full_shock_mask, unchunked.full_shock_mask)
    np.testing.assert_allclose(chunked.mach_pressure, unchunked.mach_pressure, equal_nan=True)
    np.testing.assert_allclose(chunked.mach_temperature, unchunked.mach_temperature, equal_nan=True)
    np.testing.assert_allclose(chunked.temperature_jump, unchunked.temperature_jump, equal_nan=True)
    np.testing.assert_allclose(chunked.pressure_jump, unchunked.pressure_jump, equal_nan=True)
    np.testing.assert_allclose(chunked.density_jump, unchunked.density_jump, equal_nan=True)
    np.testing.assert_allclose(chunked.normal_x, unchunked.normal_x, equal_nan=True)
    np.testing.assert_allclose(chunked.normal_y, unchunked.normal_y, equal_nan=True)
    np.testing.assert_allclose(chunked.normal_z, unchunked.normal_z, equal_nan=True)
    assert chunked.summary.keys() == unchunked.summary.keys()
    for key, chunked_value in chunked.summary.items():
        unchunked_value = unchunked.summary[key]
        if isinstance(chunked_value, float):
            np.testing.assert_allclose(chunked_value, unchunked_value, equal_nan=True)
        else:
            assert chunked_value == unchunked_value


def test_chunked_nearest_axis_matches_full_cube() -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    chunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=8,
            sampling_method="nearest_axis",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    unchunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=None,
            sampling_method="nearest_axis",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    _assert_same_result(chunked, unchunked)


def test_chunked_trilinear_matches_full_cube() -> None:
    cube = make_oblique_shock_cube(mach=2.0, n=18)
    chunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=7,
            sampling_method="trilinear",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    unchunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=None,
            sampling_method="trilinear",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    _assert_same_result(chunked, unchunked)


def test_chunked_hdf5_metadata_shape_matches_explicit_chunking() -> None:
    cube = make_planar_shock_cube(mach=2.0, n=18)
    cube.metadata["source_chunk_shape"] = (8, 8, 8)
    auto_chunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=None,
            sampling_method="nearest_axis",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    explicit_chunked = ShockFinder(
        ShockFinderConfig(
            chunk_size=8,
            sampling_method="nearest_axis",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube, progress=False)
    _assert_same_result(auto_chunked, explicit_chunked)


def test_chunk_slices_are_balanced_near_target_size() -> None:
    slices = list(iter_balanced_slices((300, 260, 129), 128))
    axis_lengths = [{slc[axis].stop - slc[axis].start for slc in slices} for axis in range(3)]

    assert axis_lengths[0] == {100}
    assert axis_lengths[1] == {86, 87}
    assert axis_lengths[2] == {64, 65}


def test_chunk_slices_support_per_axis_chunk_shapes() -> None:
    slices = list(iter_balanced_slices((18, 18, 18), (8, 6, 5)))
    axis_lengths = [{slc[axis].stop - slc[axis].start for slc in slices} for axis in range(3)]

    assert axis_lengths[0] == {6}
    assert axis_lengths[1] == {6}
    assert axis_lengths[2] == {4, 5}

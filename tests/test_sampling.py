from __future__ import annotations

import numpy as np
import pytest

from shockit import ShockFinder, ShockFinderConfig
from shockit.sampling import sample_jumps
from tests.helpers import make_oblique_shock_cube, make_planar_shock_cube


def test_nearest_axis_recovers_planar_x_shock_mach() -> None:
    cube = make_planar_shock_cube(mach=2.0, n=32)
    result = ShockFinder(
        ShockFinderConfig(
            sampling_method="nearest_axis",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube)
    recovered = result.mach_pressure[result.shock_mask]
    assert recovered.size > 0
    np.testing.assert_allclose(np.median(recovered), 2.0, rtol=1e-6, atol=1e-6)


def test_trilinear_detects_small_oblique_shock() -> None:
    cube = make_oblique_shock_cube(mach=2.0, n=20)
    result = ShockFinder(
        ShockFinderConfig(
            sampling_method="trilinear",
            reduce_to_centers=False,
            min_mach=1.1,
        )
    ).find(cube)
    assert np.count_nonzero(result.shock_mask) > 0
    assert np.isfinite(result.mach_pressure[result.shock_mask]).any() or np.isfinite(
        result.mach_temperature[result.shock_mask]
    ).any()


def test_trilinear_sampling_uses_candidate_cells_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cube = make_oblique_shock_cube(mach=2.0, n=12)
    mask = np.zeros(cube.rho.shape, dtype=bool)
    mask[4:6, 4:6, 4:6] = True

    def forbidden_meshgrid(*args, **kwargs):
        raise AssertionError("full coordinate grid allocation should not be used for candidate-only trilinear sampling")

    monkeypatch.setattr("shockit.sampling.np.meshgrid", forbidden_meshgrid)
    jumps = sample_jumps(
        pressure=cube.pressure,
        temperature=cube.pressure / cube.rho,
        rho=cube.rho,
        nx=np.ones_like(cube.rho) / np.sqrt(2.0),
        ny=np.ones_like(cube.rho) / np.sqrt(2.0),
        nz=np.zeros_like(cube.rho),
        width=1,
        method="trilinear",
        candidate_mask=mask,
    )
    assert np.isfinite(jumps.pressure_jump[mask]).any()
    assert np.isnan(jumps.pressure_jump[~mask]).all()


def test_invalid_upstream_values_produce_nan_jump_safely() -> None:
    shape = (3, 3, 3)
    pressure = np.ones(shape)
    temperature = np.ones(shape)
    rho = np.ones(shape)
    pressure[0, :, :] = 0.0
    temperature[0, :, :] = 0.0
    rho[0, :, :] = 0.0

    jumps = sample_jumps(
        pressure=pressure,
        temperature=temperature,
        rho=rho,
        nx=np.ones(shape),
        ny=np.zeros(shape),
        nz=np.zeros(shape),
        width=1,
        method="nearest_axis",
    )
    assert np.isnan(jumps.pressure_jump[1, 1, 1])
    assert np.isnan(jumps.temperature_jump[1, 1, 1])
    assert np.isnan(jumps.density_jump[1, 1, 1])

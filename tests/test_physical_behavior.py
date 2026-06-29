from __future__ import annotations

import numpy as np

from shockit import FluidCube, ShockFinder, ShockFinderConfig


def test_uniform_field_has_no_shock_cells_and_nan_mach_fields() -> None:
    cube = FluidCube(
        rho=np.ones((8, 8, 8)),
        pressure=np.ones((8, 8, 8)),
        vx=np.zeros((8, 8, 8)),
        vy=np.zeros((8, 8, 8)),
        vz=np.zeros((8, 8, 8)),
    )

    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(cube, progress=False)

    assert result.summary["shock_cells"] == 0
    assert result.summary["shock_zone_cells"] == 0
    assert result.summary["full_shock_cells"] == 0
    assert not np.any(result.shock_zone_mask)
    assert not np.any(result.full_shock_mask)
    assert not np.any(result.shock_mask)
    assert np.isnan(result.mach_pressure).all()
    assert np.isnan(result.mach_temperature).all()


def test_pressure_jump_without_compression_is_rejected() -> None:
    rho = np.ones((12, 12, 12))
    pressure = np.ones((12, 12, 12))
    pressure[6:, :, :] = 4.0
    vx = np.zeros((12, 12, 12))
    vy = np.zeros((12, 12, 12))
    vz = np.zeros((12, 12, 12))

    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(
        FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz),
        progress=False,
    )

    assert not np.any(result.full_shock_mask)
    assert result.summary["shock_cells"] == 0


def test_compression_without_thermodynamic_jump_is_rejected() -> None:
    x = np.linspace(-1.0, 1.0, 16)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    cube = FluidCube(
        rho=np.ones((16, 16, 16)),
        pressure=np.ones((16, 16, 16)),
        vx=-xx,
        vy=-yy,
        vz=-zz,
    )

    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(cube, progress=False)

    assert not np.any(result.full_shock_mask)
    assert result.summary["shock_cells"] == 0

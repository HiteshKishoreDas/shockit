import numpy as np

from shockit import FluidCube, ShockFinder, ShockFinderConfig


def test_sod_like_setup_finds_shock_more_than_contact(test_plotter) -> None:
    n = 48
    gamma = 1.4
    rho = np.ones((n, 8, 8))
    pressure = np.ones((n, 8, 8)) * 0.5
    vx = np.zeros((n, 8, 8))
    vy = np.zeros((n, 8, 8))
    vz = np.zeros((n, 8, 8))

    # First interface is a contact-like jump at nearly constant pressure.
    rho[: n // 3] = 2.0
    pressure[: n // 3] = 0.5
    rho[n // 3 : 2 * n // 3] = 1.0
    pressure[n // 3 : 2 * n // 3] = 0.5

    # Second interface is shock-like with compression and thermodynamic jumps.
    rho[2 * n // 3 :] = 2.0
    pressure[2 * n // 3 :] = 2.0

    vx[: n // 3] = 1.0
    vx[n // 3 : 2 * n // 3] = 1.0
    vx[2 * n // 3 :] = 0.4

    cube = FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz, gamma=gamma)
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.05)).find(cube)
    test_plotter.save_slice(
        "sod_pressure_midplane",
        cube.pressure[:, :, cube.pressure.shape[2] // 2],
        title="Sod-like pressure slice",
        colorbar_label="pressure",
    )
    test_plotter.save_slice(
        "sod_shock_mask_midplane",
        result.shock_mask[:, :, cube.pressure.shape[2] // 2].astype(float),
        title="Sod-like detected shock cells",
        cmap="magma",
        colorbar_label="shock mask",
    )
    test_plotter.save_line(
        "sod_pressure_mach_profile_x",
        np.arange(cube.rho.shape[0]),
        result.mach_pressure[:, cube.rho.shape[1] // 2, cube.rho.shape[2] // 2],
        title="Sod-like pressure-Mach profile",
        xlabel="x index",
        ylabel="Mach from pressure jump",
        label="Mach",
    )
    assert np.count_nonzero(result.shock_mask) > 0
    shock_positions = np.argwhere(result.shock_mask)[:, 0]
    assert shock_positions.mean() > n / 2.0

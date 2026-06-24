import numpy as np

from cube_shockfinder import FluidCube, ShockFinder, ShockFinderConfig


def test_contact_discontinuity_is_rejected(test_plotter) -> None:
    n = 32
    rho = np.ones((n, n, n))
    pressure = np.ones((n, n, n))
    vx = np.zeros((n, n, n))
    vy = np.zeros((n, n, n))
    vz = np.zeros((n, n, n))
    rho[n // 2 :, :, :] = 2.0
    cube = FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz)
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(cube)
    test_plotter.save_slice(
        "contact_density_midplane",
        cube.rho[:, :, n // 2],
        title="Contact discontinuity density slice",
        colorbar_label="rho",
    )
    test_plotter.save_slice(
        "contact_shock_mask_midplane",
        result.shock_mask[:, :, n // 2].astype(float),
        title="Contact discontinuity shock mask",
        cmap="magma",
        colorbar_label="shock mask",
    )
    assert np.count_nonzero(result.shock_mask) == 0

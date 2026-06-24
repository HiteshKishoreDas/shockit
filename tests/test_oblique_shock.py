import numpy as np

from cube_shockfinder import FluidCube, ShockFinder, ShockFinderConfig
from cube_shockfinder.mach import density_jump_from_mach, pressure_jump_from_mach


def test_oblique_shock_detected_with_reasonable_normal_alignment(test_plotter) -> None:
    n = 24
    gamma = 5.0 / 3.0
    mach = 2.0
    rho1 = 1.0
    p1 = 1.0
    rjump = density_jump_from_mach(mach, gamma)
    pjump = pressure_jump_from_mach(mach, gamma)
    rho2 = rho1 * rjump
    p2 = p1 * pjump

    x = np.arange(n)
    y = np.arange(n)
    z = np.arange(n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    signed_distance = (xx + yy) - n

    rho = np.where(signed_distance >= 0, rho2, rho1).astype(float)
    pressure = np.where(signed_distance >= 0, p2, p1).astype(float)
    vx = np.where(signed_distance >= 0, 0.5, 2.0).astype(float)
    vy = np.where(signed_distance >= 0, 0.5, 2.0).astype(float)
    vz = np.zeros((n, n, n))

    cube = FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz, gamma=gamma)
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(cube)
    test_plotter.save_slice(
        "oblique_pressure_midplane",
        cube.pressure[:, :, n // 2],
        title="Oblique shock pressure slice",
        colorbar_label="pressure",
    )
    test_plotter.save_slice(
        "oblique_shock_mask_midplane",
        result.shock_mask[:, :, n // 2].astype(float),
        title="Oblique shock detected cells",
        cmap="magma",
        colorbar_label="shock mask",
    )
    assert np.count_nonzero(result.shock_mask) > 0

    true_normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    normals = np.stack(
        [
            result.normal_x[result.shock_mask],
            result.normal_y[result.shock_mask],
            result.normal_z[result.shock_mask],
        ],
        axis=1,
    )
    alignment = normals @ true_normal
    test_plotter.save_line(
        "oblique_normal_alignment",
        np.arange(alignment.size),
        alignment,
        title="Oblique shock normal alignment",
        xlabel="Shock-cell index",
        ylabel="n . n_true",
        label="Alignment",
    )
    assert np.mean(alignment) > 0.5

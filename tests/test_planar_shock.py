import numpy as np

from shockit import FluidCube, ShockFinder, ShockFinderConfig
from tests.helpers import make_planar_shock_cube


def test_planar_shock_detected_near_interface(test_plotter) -> None:
    cube = make_planar_shock_cube()
    result = ShockFinder(ShockFinderConfig(min_mach=1.1, reduce_to_centers=False)).find(cube)
    interface_slice = cube.rho.shape[0] // 2
    nearby = result.shock_mask[max(0, interface_slice - 1) : interface_slice + 1]
    test_plotter.save_slice(
        "planar_pressure_midplane",
        cube.pressure[:, :, cube.pressure.shape[2] // 2],
        title="Planar shock pressure slice",
        colorbar_label="pressure",
    )
    test_plotter.save_slice(
        "planar_shock_mask_midplane",
        result.shock_mask[:, :, cube.pressure.shape[2] // 2].astype(float),
        title="Planar shock detected cells",
        cmap="magma",
        colorbar_label="shock mask",
    )
    test_plotter.save_line(
        "planar_mach_profile_x",
        np.arange(cube.rho.shape[0]),
        result.mach_pressure[:, cube.rho.shape[1] // 2, cube.rho.shape[2] // 2],
        title="Planar shock pressure-Mach profile",
        xlabel="x index",
        ylabel="Mach from pressure jump",
        label="Mach",
    )
    assert np.count_nonzero(nearby) > 0
    recovered = result.mach_pressure[result.shock_mask]
    np.testing.assert_allclose(np.median(recovered), 2.0, rtol=0.2)
    assert result.summary["full_shock_cells"] >= result.summary["shock_cells"]
    assert result.summary["full_mach_pressure_median"] >= 1.0

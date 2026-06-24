import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from tests.helpers import make_oblique_shock_cube


def test_oblique_shock_detected_with_reasonable_normal_alignment(test_plotter) -> None:
    cube = make_oblique_shock_cube()
    n = cube.rho.shape[0]
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

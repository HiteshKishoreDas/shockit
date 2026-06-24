import numpy as np

from cube_shockfinder.gradients import divergence, periodic_derivative


def test_periodic_derivative_matches_sine(test_plotter) -> None:
    nx = 128
    length = 1.0
    dx = length / nx
    x = np.arange(nx) * dx
    field_1d = np.sin(2.0 * np.pi * x / length)
    expected_1d = (2.0 * np.pi / length) * np.cos(2.0 * np.pi * x / length)
    field = field_1d[:, None, None] * np.ones((1, 4, 4))
    expected = expected_1d[:, None, None] * np.ones((1, 4, 4))
    numerical = periodic_derivative(field, axis=0, spacing=dx)
    error = np.max(np.abs(numerical - expected))
    assert error < 1.0e-2
    test_plotter.save_line(
        "periodic_derivative_vs_analytic",
        x,
        numerical[:, 0, 0],
        title="Periodic derivative of sin(x)",
        xlabel="x",
        ylabel="df/dx",
        label="Numerical derivative",
    )


def test_divergence_matches_analytic_result(test_plotter) -> None:
    nx = ny = nz = 64
    length = 1.0
    dx = dy = dz = length / nx
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    vx = np.sin(2.0 * np.pi * xx / length)
    vy = np.sin(2.0 * np.pi * yy / length)
    vz = np.sin(2.0 * np.pi * zz / length)
    expected = (
        (2.0 * np.pi / length) * np.cos(2.0 * np.pi * xx / length)
        + (2.0 * np.pi / length) * np.cos(2.0 * np.pi * yy / length)
        + (2.0 * np.pi / length) * np.cos(2.0 * np.pi * zz / length)
    )
    numerical = divergence(vx, vy, vz, dx, dy, dz)
    assert np.max(np.abs(numerical - expected)) < 3.5e-2
    test_plotter.save_slice(
        "divergence_midplane",
        numerical[:, :, nz // 2],
        title="Numerical divergence midplane",
        colorbar_label="div v",
    )

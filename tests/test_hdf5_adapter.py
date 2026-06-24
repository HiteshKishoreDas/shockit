import h5py
import numpy as np

from cube_shockfinder.io_hdf5 import load_fluid_cube_from_hdf5


def test_hdf5_adapter_loads_arrays(tmp_path) -> None:
    filename = tmp_path / "snapshot.h5"
    rho = np.ones((4, 4, 4))
    pressure = np.ones((4, 4, 4)) * 2.0
    vx = np.ones((4, 4, 4)) * 3.0
    vy = np.ones((4, 4, 4)) * 4.0
    vz = np.ones((4, 4, 4)) * 5.0

    with h5py.File(filename, "w") as handle:
        handle["density"] = rho
        handle["press"] = pressure
        handle["vel1"] = vx
        handle["vel2"] = vy
        handle["vel3"] = vz

    cube = load_fluid_cube_from_hdf5(str(filename))
    np.testing.assert_allclose(cube.rho, rho)
    np.testing.assert_allclose(cube.pressure, pressure)
    np.testing.assert_allclose(cube.vx, vx)
    np.testing.assert_allclose(cube.vy, vy)
    np.testing.assert_allclose(cube.vz, vz)

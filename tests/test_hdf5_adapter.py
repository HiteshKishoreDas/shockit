import h5py
import numpy as np
import pytest

from shockit.io_hdf5 import AmbiguousFieldError, load_fluid_cube_from_hdf5


def test_hdf5_adapter_loads_arrays(tmp_path) -> None:
    filename = tmp_path / "snapshot.h5"
    rho = np.ones((4, 4, 4))
    pressure = np.ones((4, 4, 4)) * 2.0
    vx = np.ones((4, 4, 4)) * 3.0
    vy = np.ones((4, 4, 4)) * 4.0
    vz = np.ones((4, 4, 4)) * 5.0

    with h5py.File(filename, "w") as handle:
        handle.create_dataset("density", data=rho, chunks=(2, 2, 2))
        handle.create_dataset("press", data=pressure, chunks=(2, 2, 2))
        handle.create_dataset("vel1", data=vx, chunks=(2, 2, 2))
        handle.create_dataset("vel2", data=vy, chunks=(2, 2, 2))
        handle.create_dataset("vel3", data=vz, chunks=(2, 2, 2))

    cube = load_fluid_cube_from_hdf5(str(filename))
    np.testing.assert_allclose(cube.rho, rho)
    np.testing.assert_allclose(cube.pressure, pressure)
    np.testing.assert_allclose(cube.vx, vx)
    np.testing.assert_allclose(cube.vy, vy)
    np.testing.assert_allclose(cube.vz, vz)
    assert cube.metadata["rho_field"] == "density"
    assert cube.metadata["pressure_field"] == "press"
    assert cube.metadata["source_chunk_shape"] == (2, 2, 2)
    assert cube.metadata["field_chunk_shapes"]["rho"] == (2, 2, 2)


def test_hdf5_adapter_supports_nested_dataset_paths(tmp_path) -> None:
    filename = tmp_path / "snapshot_nested.h5"
    rho = np.ones((4, 4, 4))
    pressure = np.ones((4, 4, 4)) * 2.0
    vx = np.ones((4, 4, 4)) * 3.0
    vy = np.ones((4, 4, 4)) * 4.0
    vz = np.ones((4, 4, 4)) * 5.0

    with h5py.File(filename, "w") as handle:
        prim = handle.create_group("prim")
        prim["rho"] = rho
        prim["press"] = pressure
        vel = handle.create_group("vel")
        vel["vx"] = vx
        vel["vy"] = vy
        vel["vz"] = vz

    cube = load_fluid_cube_from_hdf5(
        str(filename),
        rho_field="prim/rho",
        pressure_field="prim/press",
        vx_field="vel/vx",
        vy_field="vel/vy",
        vz_field="vel/vz",
    )
    np.testing.assert_allclose(cube.rho, rho)
    np.testing.assert_allclose(cube.pressure, pressure)
    assert cube.metadata["rho_field"] == "prim/rho"
    assert cube.metadata["pressure_field"] == "prim/press"


def test_hdf5_adapter_uses_unique_nested_basename_fallback(tmp_path) -> None:
    filename = tmp_path / "snapshot_unique_nested.h5"
    rho = np.ones((4, 4, 4))
    pressure = np.ones((4, 4, 4)) * 2.0
    vx = np.ones((4, 4, 4)) * 3.0
    vy = np.ones((4, 4, 4)) * 4.0
    vz = np.ones((4, 4, 4)) * 5.0

    with h5py.File(filename, "w") as handle:
        hydro = handle.create_group("hydro")
        hydro["rho"] = rho
        hydro["pressure"] = pressure
        velocity = handle.create_group("velocity")
        velocity["vx"] = vx
        velocity["vy"] = vy
        velocity["vz"] = vz

    cube = load_fluid_cube_from_hdf5(str(filename))
    assert cube.metadata["rho_field"] == "hydro/rho"
    assert cube.metadata["pressure_field"] == "hydro/pressure"
    assert cube.metadata["vx_field"] == "velocity/vx"


def test_hdf5_adapter_raises_on_ambiguous_basename_fallback(tmp_path) -> None:
    filename = tmp_path / "snapshot_ambiguous.h5"
    array = np.ones((2, 2, 2))

    with h5py.File(filename, "w") as handle:
        handle.create_group("prim")["rho"] = array
        handle.create_group("cons")["rho"] = array * 2.0
        handle["pressure"] = array
        handle["vx"] = array
        handle["vy"] = array
        handle["vz"] = array

    with pytest.raises(AmbiguousFieldError, match="(prim/rho.*cons/rho|cons/rho.*prim/rho)"):
        load_fluid_cube_from_hdf5(str(filename))


def test_hdf5_adapter_omits_shared_chunk_shape_when_fields_do_not_match(tmp_path) -> None:
    filename = tmp_path / "snapshot_mixed_chunks.h5"
    array = np.ones((4, 4, 4))

    with h5py.File(filename, "w") as handle:
        handle.create_dataset("rho", data=array, chunks=(2, 2, 2))
        handle.create_dataset("pressure", data=array, chunks=(4, 2, 2))
        handle.create_dataset("vx", data=array, chunks=(2, 2, 2))
        handle.create_dataset("vy", data=array, chunks=(2, 2, 2))
        handle.create_dataset("vz", data=array, chunks=(2, 2, 2))

    cube = load_fluid_cube_from_hdf5(str(filename))
    assert cube.metadata["source_chunk_shape"] is None
    assert cube.metadata["field_chunk_shapes"]["pressure"] == (4, 2, 2)

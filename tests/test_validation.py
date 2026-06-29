import math

import numpy as np
import pytest

from shockit import FluidCube, ShockFinderConfig
from shockit.chunking import save_chunked_field
from shockit.storage import ChunkedFieldReference


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_mach": 0.9}, "min_mach"),
        ({"grad_T_min": -1.0}, "grad_T_min"),
        ({"shock_width_cells": 0}, "shock_width_cells"),
        ({"center_score": "bad"}, "center_score"),
        ({"sampling_method": "bad"}, "sampling_method"),
        ({"upstream_pressure_floor": -1.0}, "upstream_pressure_floor"),
        ({"upstream_temperature_floor": -1.0}, "upstream_temperature_floor"),
        ({"upstream_density_floor": -1.0}, "upstream_density_floor"),
    ],
)
def test_config_validation(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShockFinderConfig(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    ["rho", "pressure", "vx", "vy", "vz"],
)
def test_fluid_cube_rejects_non_finite_fields(field_name: str) -> None:
    fields = {
        "rho": np.ones((2, 2, 2)),
        "pressure": np.ones((2, 2, 2)),
        "vx": np.zeros((2, 2, 2)),
        "vy": np.zeros((2, 2, 2)),
        "vz": np.zeros((2, 2, 2)),
    }
    fields[field_name][0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        FluidCube(**fields)


def test_fluid_cube_rejects_non_finite_spacing() -> None:
    rho = np.ones((2, 2, 2))
    pressure = np.ones((2, 2, 2))
    velocity = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="finite"):
        FluidCube(rho=rho, pressure=pressure, vx=velocity, vy=velocity, vz=velocity, dx=math.inf)


def test_fluid_cube_rejects_npz_archive_inputs(tmp_path) -> None:
    archive_path = tmp_path / "rho.npz"
    np.savez(archive_path, arr_0=np.ones((2, 2, 2)))
    rho_archive = np.load(archive_path)
    pressure = np.ones((2, 2, 2))
    velocity = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match=r'npz archive.*arr_0'):
        FluidCube(rho=rho_archive, pressure=pressure, vx=velocity, vy=velocity, vz=velocity)


def test_fluid_cube_reports_field_name_for_bad_array_conversion() -> None:
    pressure = np.ones((2, 2, 2))
    velocity = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="Field rho could not be converted"):
        FluidCube(rho={"bad": "input"}, pressure=pressure, vx=velocity, vy=velocity, vz=velocity)


def test_fluid_cube_detects_in_memory_mode() -> None:
    cube = FluidCube(
        rho=np.ones((2, 2, 2)),
        pressure=np.ones((2, 2, 2)),
        vx=np.zeros((2, 2, 2)),
        vy=np.zeros((2, 2, 2)),
        vz=np.zeros((2, 2, 2)),
    )

    assert cube.is_in_memory
    assert not cube.is_chunked
    assert cube.storage_mode == "in_memory"
    assert cube.shape == (2, 2, 2)
    assert cube.rho.dtype == float
    assert cube.pressure.dtype == float
    assert cube.vx.dtype == float


def test_fluid_cube_converts_in_memory_fields_to_float_arrays() -> None:
    cube = FluidCube(
        rho=np.ones((2, 2, 2), dtype=np.int16),
        pressure=np.full((2, 2, 2), 2, dtype=np.int32),
        vx=np.zeros((2, 2, 2), dtype=np.int8),
        vy=np.zeros((2, 2, 2), dtype=np.int8),
        vz=np.zeros((2, 2, 2), dtype=np.int8),
    )

    assert cube.rho.dtype == float
    assert cube.pressure.dtype == float
    assert cube.vx.dtype == float


def test_fluid_cube_detects_chunked_mode(tmp_path) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)

    cube = FluidCube(
        rho=root / "rho",
        pressure=root / "prs",
        vx=root / "v1",
        vy=root / "v2",
        vz=root / "v3",
    )

    assert cube.is_chunked
    assert not cube.is_in_memory
    assert cube.storage_mode == "chunked"
    assert isinstance(cube.rho, ChunkedFieldReference)
    assert cube.shape == (4, 4, 4)


def test_fluid_cube_chunked_reference_loads_original_values(tmp_path) -> None:
    root = tmp_path / "chunked"
    rho = np.arange(64, dtype=float).reshape(4, 4, 4)
    pressure = np.full((4, 4, 4), 2.0)
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", rho, root, target_chunk_size=2)
    save_chunked_field("prs", pressure, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)

    cube = FluidCube(
        rho=root / "rho",
        pressure=root / "prs",
        vx=root / "v1",
        vy=root / "v2",
        vz=root / "v3",
    )

    np.testing.assert_array_equal(cube.rho.load(), rho)


@pytest.mark.parametrize("field_name", ["dx", "dy", "dz"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, math.nan])
def test_fluid_cube_rejects_bad_scalar_metadata_in_memory(field_name: str, bad_value: float) -> None:
    kwargs = {"dx": 1.0, "dy": 1.0, "dz": 1.0, "gamma": 5.0 / 3.0}
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name if not math.isfinite(bad_value) else "strictly positive"):
        FluidCube(
            rho=np.ones((2, 2, 2)),
            pressure=np.ones((2, 2, 2)),
            vx=np.zeros((2, 2, 2)),
            vy=np.zeros((2, 2, 2)),
            vz=np.zeros((2, 2, 2)),
            **kwargs,
        )


@pytest.mark.parametrize("bad_gamma", [1.0, 0.9, math.inf, math.nan])
def test_fluid_cube_rejects_bad_gamma_in_memory(bad_gamma: float) -> None:
    match = "greater than 1" if math.isfinite(bad_gamma) else "gamma must be finite"
    with pytest.raises(ValueError, match=match):
        FluidCube(
            rho=np.ones((2, 2, 2)),
            pressure=np.ones((2, 2, 2)),
            vx=np.zeros((2, 2, 2)),
            vy=np.zeros((2, 2, 2)),
            vz=np.zeros((2, 2, 2)),
            gamma=bad_gamma,
        )


@pytest.mark.parametrize("field_name", ["dx", "dy", "dz"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, math.nan])
def test_fluid_cube_rejects_bad_scalar_metadata_in_chunked_mode(tmp_path, field_name: str, bad_value: float) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)
    kwargs = {"dx": 1.0, "dy": 1.0, "dz": 1.0, "gamma": 5.0 / 3.0}
    kwargs[field_name] = bad_value

    match = "must be finite" if not math.isfinite(bad_value) else "strictly positive"
    with pytest.raises(ValueError, match=match):
        FluidCube(
            rho=root / "rho",
            pressure=root / "prs",
            vx=root / "v1",
            vy=root / "v2",
            vz=root / "v3",
            **kwargs,
        )


@pytest.mark.parametrize("bad_gamma", [1.0, 0.9, math.inf, math.nan])
def test_fluid_cube_rejects_bad_gamma_in_chunked_mode(tmp_path, bad_gamma: float) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)
    match = "greater than 1" if math.isfinite(bad_gamma) else "gamma must be finite"

    with pytest.raises(ValueError, match=match):
        FluidCube(
            rho=root / "rho",
            pressure=root / "prs",
            vx=root / "v1",
            vy=root / "v2",
            vz=root / "v3",
            gamma=bad_gamma,
        )


def test_fluid_cube_rejects_mixed_array_and_path_fields(tmp_path) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)

    with pytest.raises(ValueError, match="either all in-memory arrays or all chunked field paths"):
        FluidCube(
            rho=np.ones((4, 4, 4)),
            pressure=root / "prs",
            vx=root / "v1",
            vy=root / "v2",
            vz=root / "v3",
        )


def test_fluid_cube_chunked_validation_rejects_missing_directory(tmp_path) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", values, root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)

    with pytest.raises(FileNotFoundError, match="Chunked field directory"):
        FluidCube(
            rho=root / "rho",
            pressure=root / "prs",
            vx=root / "v1",
            vy=root / "v2",
            vz=root / "v3",
        )


def test_fluid_cube_chunked_validation_rejects_incompatible_layouts(tmp_path) -> None:
    root = tmp_path / "chunked"
    values = np.ones((4, 4, 4))
    zeros = np.zeros((4, 4, 4))
    save_chunked_field("rho", values, root, target_chunk_size=2)
    save_chunked_field("prs", np.ones((6, 4, 4)), root, target_chunk_size=2)
    save_chunked_field("v1", zeros, root, target_chunk_size=2)
    save_chunked_field("v2", zeros, root, target_chunk_size=2)
    save_chunked_field("v3", zeros, root, target_chunk_size=2)

    with pytest.raises(ValueError, match="layout incompatible"):
        FluidCube(
            rho=root / "rho",
            pressure=root / "prs",
            vx=root / "v1",
            vy=root / "v2",
            vz=root / "v3",
        )

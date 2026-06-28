import math

import numpy as np
import pytest

from shockit import FluidCube, ShockFinderConfig


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

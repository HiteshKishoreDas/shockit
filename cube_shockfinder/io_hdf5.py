"""HDF5 adapter for loading FluidCube instances."""

from __future__ import annotations

from collections.abc import Sequence

import h5py
import numpy as np

from .fields import FluidCube

ALIASES = {
    "rho": ("rho", "density", "dens"),
    "pressure": ("pressure", "press", "prs"),
    "vx": ("vx", "vel1", "velocity_x"),
    "vy": ("vy", "vel2", "velocity_y"),
    "vz": ("vz", "vel3", "velocity_z"),
}


def _resolve_field_name(handle: h5py.File, explicit_name: str, aliases: Sequence[str]) -> str:
    if explicit_name in handle:
        return explicit_name
    for alias in aliases:
        if alias in handle:
            return alias
    raise KeyError(f"Could not find any of these fields: {', '.join((explicit_name, *aliases))}")


def load_fluid_cube_from_hdf5(
    filename: str,
    rho_field: str = "rho",
    pressure_field: str = "pressure",
    vx_field: str = "vx",
    vy_field: str = "vy",
    vz_field: str = "vz",
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    gamma: float = 5.0 / 3.0,
) -> FluidCube:
    """Load a FluidCube from top-level HDF5 datasets."""

    with h5py.File(filename, "r") as handle:
        rho_name = _resolve_field_name(handle, rho_field, ALIASES["rho"])
        pressure_name = _resolve_field_name(handle, pressure_field, ALIASES["pressure"])
        vx_name = _resolve_field_name(handle, vx_field, ALIASES["vx"])
        vy_name = _resolve_field_name(handle, vy_field, ALIASES["vy"])
        vz_name = _resolve_field_name(handle, vz_field, ALIASES["vz"])
        return FluidCube(
            rho=np.asarray(handle[rho_name]),
            pressure=np.asarray(handle[pressure_name]),
            vx=np.asarray(handle[vx_name]),
            vy=np.asarray(handle[vy_name]),
            vz=np.asarray(handle[vz_name]),
            dx=dx,
            dy=dy,
            dz=dz,
            gamma=gamma,
            metadata={"source_file": filename},
        )

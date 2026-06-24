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


class AmbiguousFieldError(KeyError):
    """Raised when recursive basename resolution finds multiple datasets."""


def _dataset_exists(handle: h5py.File, path: str) -> bool:
    try:
        return isinstance(handle[path], h5py.Dataset)
    except KeyError:
        return False


def _find_datasets_by_basename(handle: h5py.File, basename: str) -> list[str]:
    found: list[str] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and name.rsplit("/", maxsplit=1)[-1] == basename:
            found.append(name)

    handle.visititems(visitor)
    return found


def _resolve_field_name(handle: h5py.File, explicit_name: str, aliases: Sequence[str]) -> str:
    if _dataset_exists(handle, explicit_name):
        return explicit_name
    for alias in aliases:
        if _dataset_exists(handle, alias):
            return alias
    for candidate in (explicit_name, *aliases):
        matches = _find_datasets_by_basename(handle, candidate)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            match_list = ", ".join(matches)
            raise AmbiguousFieldError(
                f"Ambiguous dataset basename '{candidate}' matched multiple paths: {match_list}. "
                "Pass an explicit dataset path such as 'group/field'."
            )
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
    """Load a FluidCube from HDF5 datasets, including nested dataset paths."""

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
            metadata={
                "source_file": filename,
                "rho_field": rho_name,
                "pressure_field": pressure_name,
                "vx_field": vx_name,
                "vy_field": vy_name,
                "vz_field": vz_name,
            },
        )

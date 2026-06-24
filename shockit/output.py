"""Output helpers for shock-finder results."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import h5py
import numpy as np

from .finder import ShockFinderResult


def _package_version() -> str:
    try:
        return version("shockit")
    except PackageNotFoundError:  # pragma: no cover - editable fallback
        return "0+unknown"


def _write_mask_dataset(handle: h5py.File, name: str, mask: np.ndarray) -> None:
    dataset = handle.create_dataset(name, data=mask.astype(bool), dtype=np.bool_)
    dataset.attrs["semantic_type"] = "boolean_mask"
    dataset.attrs["stored_dtype"] = str(dataset.dtype)


def save_result_hdf5(filename: str, result: ShockFinderResult) -> None:
    """Persist shock-finder outputs to an HDF5 file."""

    with h5py.File(filename, "w") as handle:
        _write_mask_dataset(handle, "shock_mask", result.shock_mask)
        _write_mask_dataset(handle, "shock_zone_mask", result.shock_zone_mask)
        if result.full_shock_mask is not None:
            _write_mask_dataset(handle, "full_shock_mask", result.full_shock_mask)
        handle.create_dataset("mach_temperature", data=result.mach_temperature)
        handle.create_dataset("mach_pressure", data=result.mach_pressure)
        handle.create_dataset("compression", data=result.compression)
        handle.create_dataset("div_v", data=result.div_v)
        handle.create_dataset("temperature", data=result.temperature)
        handle.create_dataset("entropy", data=result.entropy)
        handle.create_dataset("temperature_jump", data=result.temperature_jump)
        handle.create_dataset("pressure_jump", data=result.pressure_jump)
        handle.create_dataset("density_jump", data=result.density_jump)
        handle.create_dataset("normal_x", data=result.normal_x)
        handle.create_dataset("normal_y", data=result.normal_y)
        handle.create_dataset("normal_z", data=result.normal_z)
        handle.attrs["package_name"] = "shockit"
        handle.attrs["package_version"] = _package_version()
        handle.attrs["mach_field_note"] = "Mach fields are local cell estimates from sampled jump conditions."
        for key, value in result.summary.items():
            if isinstance(value, tuple):
                value = np.asarray(value)
            handle.attrs[key] = value

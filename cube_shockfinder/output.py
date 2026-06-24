"""Output helpers for shock-finder results."""

from __future__ import annotations

import h5py
import numpy as np

from .finder import ShockFinderResult


def save_result_hdf5(filename: str, result: ShockFinderResult) -> None:
    """Persist shock-finder outputs to an HDF5 file."""

    with h5py.File(filename, "w") as handle:
        handle.create_dataset("shock_mask", data=result.shock_mask.astype(np.uint8))
        handle.create_dataset("shock_zone_mask", data=result.shock_zone_mask.astype(np.uint8))
        if result.full_shock_mask is not None:
            handle.create_dataset("full_shock_mask", data=result.full_shock_mask.astype(np.uint8))
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
        for key, value in result.summary.items():
            if isinstance(value, tuple):
                value = np.asarray(value)
            handle.attrs[key] = value

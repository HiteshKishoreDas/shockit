"""Summary helpers for shock-finder results."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .config import ShockFinderConfig
from .derived import MachFields
from .fields import FluidCube


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    """Return finite min/median/max statistics, or NaN if none are finite."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))


def make_summary(
    cube: FluidCube,
    shock_zone_mask: np.ndarray,
    full_shock_mask: np.ndarray,
    shock_mask: np.ndarray,
    mach_fields: MachFields,
    config: ShockFinderConfig,
) -> dict[str, Any]:
    """Summarize both unreduced and final mask populations."""

    total_cells = int(np.prod(cube.shape))
    shock_cells = int(np.count_nonzero(shock_mask))
    full_shock_cells = int(np.count_nonzero(full_shock_mask))
    _, n_connected_components = ndimage.label(full_shock_mask)
    pmin, pmed, pmax = finite_stats(mach_fields.mach_pressure[shock_mask])
    tmin, tmed, tmax = finite_stats(mach_fields.mach_temperature[shock_mask])
    fpmin, fpmed, fpmax = finite_stats(mach_fields.mach_pressure[full_shock_mask])
    ftmin, ftmed, ftmax = finite_stats(mach_fields.mach_temperature[full_shock_mask])
    return {
        "grid_shape": cube.shape,
        "total_cells": total_cells,
        "shock_zone_cells": int(np.count_nonzero(shock_zone_mask)),
        "shock_cells": shock_cells,
        "shock_fraction": shock_cells / total_cells,
        "full_shock_cells": full_shock_cells,
        "full_shock_fraction": full_shock_cells / total_cells,
        "n_connected_components": int(n_connected_components),
        "mask_semantics": "shock_mask may be center-reduced; full_shock_mask is unreduced",
        "mach_pressure_min": pmin,
        "mach_pressure_median": pmed,
        "mach_pressure_max": pmax,
        "mach_temperature_min": tmin,
        "mach_temperature_median": tmed,
        "mach_temperature_max": tmax,
        "full_mach_pressure_min": fpmin,
        "full_mach_pressure_median": fpmed,
        "full_mach_pressure_max": fpmax,
        "full_mach_temperature_min": ftmin,
        "full_mach_temperature_median": ftmed,
        "full_mach_temperature_max": ftmax,
        "min_mach": config.min_mach,
        "gamma": cube.gamma,
        "shock_width_cells": config.shock_width_cells,
        "reduce_to_centers": config.reduce_to_centers,
        "center_score": config.center_score,
        "sampling_method": config.sampling_method,
    }

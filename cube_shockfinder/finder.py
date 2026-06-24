"""Main shock finding API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .criteria import aligned_dot
from .fields import FluidCube
from .gradients import divergence, gradient, magnitude
from .mach import mach_from_temperature_jump
from .sampling import sample_jumps


@dataclass
class ShockFinderConfig:
    """Configuration for the uniform-grid shock finder."""

    div_v_threshold: float = 0.0
    grad_T_min: float = 1e-30
    require_gradT_gradS_alignment: bool = True
    require_gradT_gradRho_alignment: bool = True
    normal_field: str = "temperature"
    shock_width_cells: int = 1
    require_pressure_jump: bool = True
    require_temperature_jump: bool = True
    require_density_jump: bool = True
    min_mach: float = 1.1
    mach_max: float = 1000.0
    reduce_to_centers: bool = True

    def __post_init__(self) -> None:
        if self.normal_field not in {"temperature", "pressure"}:
            raise ValueError("normal_field must be 'temperature' or 'pressure'.")
        if self.shock_width_cells < 1:
            raise ValueError("shock_width_cells must be >= 1.")
        if self.mach_max <= 1.0:
            raise ValueError("mach_max must be > 1.")


@dataclass
class ShockFinderResult:
    """Results returned by ShockFinder."""

    shock_mask: np.ndarray
    shock_zone_mask: np.ndarray
    mach_temperature: np.ndarray
    mach_pressure: np.ndarray
    compression: np.ndarray
    div_v: np.ndarray
    temperature: np.ndarray
    entropy: np.ndarray
    temperature_jump: np.ndarray
    pressure_jump: np.ndarray
    density_jump: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    normal_z: np.ndarray
    summary: dict[str, Any]
    full_shock_mask: np.ndarray | None = None


class ShockFinder:
    """Skillman-style shock finder for uniform fluid cubes."""

    def __init__(self, config: ShockFinderConfig | None = None) -> None:
        self.config = config or ShockFinderConfig()

    def find(self, cube: FluidCube) -> ShockFinderResult:
        """Run the shock finder on a fluid cube."""

        temperature = cube.pressure / cube.rho
        entropy = np.log(cube.pressure / np.power(cube.rho, cube.gamma))

        div_v = divergence(cube.vx, cube.vy, cube.vz, cube.dx, cube.dy, cube.dz)
        grad_tx, grad_ty, grad_tz = gradient(temperature, cube.dx, cube.dy, cube.dz)
        grad_sx, grad_sy, grad_sz = gradient(entropy, cube.dx, cube.dy, cube.dz)
        grad_px, grad_py, grad_pz = gradient(cube.pressure, cube.dx, cube.dy, cube.dz)
        grad_rx, grad_ry, grad_rz = gradient(cube.rho, cube.dx, cube.dy, cube.dz)

        grad_t_mag = magnitude(grad_tx, grad_ty, grad_tz)
        compression = -div_v

        shock_zone_mask = div_v < self.config.div_v_threshold
        shock_zone_mask &= grad_t_mag > self.config.grad_T_min
        if self.config.require_gradT_gradS_alignment:
            shock_zone_mask &= aligned_dot(grad_tx, grad_ty, grad_tz, grad_sx, grad_sy, grad_sz) > 0.0
        if self.config.require_gradT_gradRho_alignment:
            shock_zone_mask &= aligned_dot(grad_tx, grad_ty, grad_tz, grad_rx, grad_ry, grad_rz) > 0.0

        if self.config.normal_field == "temperature":
            normal_x_raw, normal_y_raw, normal_z_raw = grad_tx, grad_ty, grad_tz
            normal_mag = grad_t_mag
        else:
            normal_x_raw, normal_y_raw, normal_z_raw = grad_px, grad_py, grad_pz
            normal_mag = magnitude(grad_px, grad_py, grad_pz)

        safe_mag = np.where(normal_mag > 0.0, normal_mag, 1.0)
        normal_x = normal_x_raw / safe_mag
        normal_y = normal_y_raw / safe_mag
        normal_z = normal_z_raw / safe_mag

        jumps = sample_jumps(
            pressure=cube.pressure,
            temperature=temperature,
            rho=cube.rho,
            nx=normal_x,
            ny=normal_y,
            nz=normal_z,
            width=self.config.shock_width_cells,
        )

        mach_pressure = np.full(cube.rho.shape, np.nan, dtype=float)
        valid_pressure = jumps.pressure_jump > 1.0
        if np.any(valid_pressure):
            mach_pressure[valid_pressure] = np.sqrt(
                ((jumps.pressure_jump[valid_pressure] * (cube.gamma + 1.0)) + (cube.gamma - 1.0))
                / (2.0 * cube.gamma)
            )

        mach_temperature = np.full(cube.rho.shape, np.nan, dtype=float)
        valid_temperature = jumps.temperature_jump > 1.0
        if np.any(valid_temperature):
            values = [
                mach_from_temperature_jump(val, cube.gamma, mach_max=self.config.mach_max)
                for val in jumps.temperature_jump[valid_temperature]
            ]
            mach_temperature[valid_temperature] = np.asarray(values, dtype=float)

        jump_mask = np.ones(cube.rho.shape, dtype=bool)
        if self.config.require_pressure_jump:
            jump_mask &= jumps.pressure_jump > 1.0
        if self.config.require_temperature_jump:
            jump_mask &= jumps.temperature_jump > 1.0
        if self.config.require_density_jump:
            jump_mask &= jumps.density_jump > 1.0

        mach_mask = np.nan_to_num(mach_pressure, nan=-np.inf) >= self.config.min_mach
        mach_mask |= np.nan_to_num(mach_temperature, nan=-np.inf) >= self.config.min_mach

        full_shock_mask = shock_zone_mask & jump_mask & mach_mask
        shock_mask = full_shock_mask.copy()
        if self.config.reduce_to_centers:
            shock_mask = self._reduce_to_centers(shock_mask, div_v)

        summary = _make_summary(
            shape=cube.rho.shape,
            shock_zone_mask=shock_zone_mask,
            shock_mask=shock_mask,
            mach_pressure=mach_pressure,
            mach_temperature=mach_temperature,
            config=self.config,
            gamma=cube.gamma,
        )

        return ShockFinderResult(
            shock_mask=shock_mask,
            shock_zone_mask=shock_zone_mask,
            mach_temperature=mach_temperature,
            mach_pressure=mach_pressure,
            compression=compression,
            div_v=div_v,
            temperature=temperature,
            entropy=entropy,
            temperature_jump=jumps.temperature_jump,
            pressure_jump=jumps.pressure_jump,
            density_jump=jumps.density_jump,
            normal_x=normal_x,
            normal_y=normal_y,
            normal_z=normal_z,
            summary=summary,
            full_shock_mask=full_shock_mask,
        )

    def _reduce_to_centers(self, shock_mask: np.ndarray, div_v: np.ndarray) -> np.ndarray:
        """Reduce connected shock regions to their most compressive cells."""

        labels, count = ndimage.label(shock_mask)
        if count == 0:
            return shock_mask

        reduced = np.zeros_like(shock_mask, dtype=bool)
        for label in range(1, count + 1):
            component = labels == label
            if not np.any(component):
                continue
            component_div = np.where(component, div_v, np.inf)
            min_index = np.unravel_index(np.argmin(component_div), component_div.shape)
            reduced[min_index] = True
        return reduced


def _finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))


def _make_summary(
    shape: tuple[int, int, int],
    shock_zone_mask: np.ndarray,
    shock_mask: np.ndarray,
    mach_pressure: np.ndarray,
    mach_temperature: np.ndarray,
    config: ShockFinderConfig,
    gamma: float,
) -> dict[str, Any]:
    total_cells = int(np.prod(shape))
    pmin, pmed, pmax = _finite_stats(mach_pressure[shock_mask])
    tmin, tmed, tmax = _finite_stats(mach_temperature[shock_mask])
    shock_cells = int(np.count_nonzero(shock_mask))
    return {
        "grid_shape": shape,
        "total_cells": total_cells,
        "shock_zone_cells": int(np.count_nonzero(shock_zone_mask)),
        "shock_cells": shock_cells,
        "shock_fraction": shock_cells / total_cells,
        "mach_pressure_min": pmin,
        "mach_pressure_median": pmed,
        "mach_pressure_max": pmax,
        "mach_temperature_min": tmin,
        "mach_temperature_median": tmed,
        "mach_temperature_max": tmax,
        "min_mach": config.min_mach,
        "gamma": gamma,
        "shock_width_cells": config.shock_width_cells,
        "reduce_to_centers": config.reduce_to_centers,
    }

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
from .sampling import SampledJumps, sample_jumps


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
    center_score: str = "compression"
    sampling_method: str = "nearest_axis"

    def __post_init__(self) -> None:
        """Validate supported configuration values."""

        if self.normal_field not in {"temperature", "pressure"}:
            raise ValueError("normal_field must be 'temperature' or 'pressure'.")
        if self.grad_T_min < 0.0:
            raise ValueError("grad_T_min must be >= 0.")
        if self.min_mach < 1.0:
            raise ValueError("min_mach must be >= 1.")
        if self.shock_width_cells < 1:
            raise ValueError("shock_width_cells must be >= 1.")
        if self.mach_max <= 1.0:
            raise ValueError("mach_max must be > 1.")
        if self.center_score not in {"compression", "mach_pressure", "mach_temperature", "combined"}:
            raise ValueError("center_score must be one of: compression, mach_pressure, mach_temperature, combined.")
        if self.sampling_method not in {"nearest_axis", "trilinear"}:
            raise ValueError("sampling_method must be either 'nearest_axis' or 'trilinear'.")


@dataclass
class ShockFinderResult:
    """Results returned by ShockFinder.

    `shock_zone_mask` marks cells that satisfy the local compression and
    gradient-based shock-zone criteria.
    `full_shock_mask` marks all cells that pass jump filtering and minimum Mach.
    `shock_mask` is the final public mask, optionally center-reduced to one cell
    per connected shock region for easier cataloging.
    """

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


@dataclass
class DerivedFields:
    """Thermodynamic proxy fields used by the shock finder.

    The temperature proxy is `pressure / rho`, proportional to the ideal-gas
    temperature up to a constant factor.
    The entropy proxy is `log(pressure / rho**gamma)`, which helps separate
    shocks from contact-like jumps.
    """

    temperature: np.ndarray
    entropy: np.ndarray
    div_v: np.ndarray
    compression: np.ndarray


@dataclass
class GradientFields:
    """Spatial gradients of the thermodynamic fields and primitives."""

    grad_tx: np.ndarray
    grad_ty: np.ndarray
    grad_tz: np.ndarray
    grad_sx: np.ndarray
    grad_sy: np.ndarray
    grad_sz: np.ndarray
    grad_px: np.ndarray
    grad_py: np.ndarray
    grad_pz: np.ndarray
    grad_rx: np.ndarray
    grad_ry: np.ndarray
    grad_rz: np.ndarray
    grad_t_mag: np.ndarray
    grad_p_mag: np.ndarray


@dataclass
class NormalFields:
    """Normalized shock-normal field components."""

    normal_x: np.ndarray
    normal_y: np.ndarray
    normal_z: np.ndarray


@dataclass
class MachFields:
    """Mach fields inferred from pressure and temperature jumps."""

    mach_pressure: np.ndarray
    mach_temperature: np.ndarray


class ShockFinder:
    """Skillman-style shock finder for uniform fluid cubes."""

    def __init__(self, config: ShockFinderConfig | None = None) -> None:
        self.config = config or ShockFinderConfig()

    def find(self, cube: FluidCube) -> ShockFinderResult:
        """Run the shock finder on a fluid cube."""

        derived = self._compute_derived_fields(cube)
        gradients = self._compute_gradients(cube, derived.temperature, derived.entropy)
        shock_zone_mask = self._build_shock_zone_mask(derived, gradients)
        normals = self._compute_normals(gradients)
        jumps = sample_jumps(
            pressure=cube.pressure,
            temperature=derived.temperature,
            rho=cube.rho,
            nx=normals.normal_x,
            ny=normals.normal_y,
            nz=normals.normal_z,
            width=self.config.shock_width_cells,
            method=self.config.sampling_method,
        )
        mach_fields = self._compute_mach_fields(jumps, cube.gamma)
        full_shock_mask = self._build_final_mask(shock_zone_mask, jumps, mach_fields)
        shock_mask = full_shock_mask.copy()
        if self.config.reduce_to_centers:
            shock_mask = self._reduce_to_centers(full_shock_mask, derived.div_v, mach_fields)

        summary = self._make_summary(
            cube=cube,
            shock_zone_mask=shock_zone_mask,
            shock_mask=shock_mask,
            full_shock_mask=full_shock_mask,
            mach_fields=mach_fields,
        )
        return ShockFinderResult(
            shock_mask=shock_mask,
            shock_zone_mask=shock_zone_mask,
            mach_temperature=mach_fields.mach_temperature,
            mach_pressure=mach_fields.mach_pressure,
            compression=derived.compression,
            div_v=derived.div_v,
            temperature=derived.temperature,
            entropy=derived.entropy,
            temperature_jump=jumps.temperature_jump,
            pressure_jump=jumps.pressure_jump,
            density_jump=jumps.density_jump,
            normal_x=normals.normal_x,
            normal_y=normals.normal_y,
            normal_z=normals.normal_z,
            summary=summary,
            full_shock_mask=full_shock_mask,
        )

    def _compute_derived_fields(self, cube: FluidCube) -> DerivedFields:
        """Compute temperature, entropy, divergence, and compression proxies."""

        temperature = cube.pressure / cube.rho
        entropy = np.log(cube.pressure / np.power(cube.rho, cube.gamma))
        div_v = divergence(cube.vx, cube.vy, cube.vz, cube.dx, cube.dy, cube.dz)
        compression = -div_v
        return DerivedFields(
            temperature=temperature,
            entropy=entropy,
            div_v=div_v,
            compression=compression,
        )

    def _compute_gradients(self, cube: FluidCube, temperature: np.ndarray, entropy: np.ndarray) -> GradientFields:
        """Compute periodic centered gradients of the relevant fields."""

        grad_tx, grad_ty, grad_tz = gradient(temperature, cube.dx, cube.dy, cube.dz)
        grad_sx, grad_sy, grad_sz = gradient(entropy, cube.dx, cube.dy, cube.dz)
        grad_px, grad_py, grad_pz = gradient(cube.pressure, cube.dx, cube.dy, cube.dz)
        grad_rx, grad_ry, grad_rz = gradient(cube.rho, cube.dx, cube.dy, cube.dz)
        return GradientFields(
            grad_tx=grad_tx,
            grad_ty=grad_ty,
            grad_tz=grad_tz,
            grad_sx=grad_sx,
            grad_sy=grad_sy,
            grad_sz=grad_sz,
            grad_px=grad_px,
            grad_py=grad_py,
            grad_pz=grad_pz,
            grad_rx=grad_rx,
            grad_ry=grad_ry,
            grad_rz=grad_rz,
            grad_t_mag=magnitude(grad_tx, grad_ty, grad_tz),
            grad_p_mag=magnitude(grad_px, grad_py, grad_pz),
        )

    def _build_shock_zone_mask(self, derived: DerivedFields, gradients: GradientFields) -> np.ndarray:
        """Build the preliminary shock-zone mask from compression and alignment criteria."""

        mask = derived.div_v < self.config.div_v_threshold
        mask &= gradients.grad_t_mag > self.config.grad_T_min
        if self.config.require_gradT_gradS_alignment:
            mask &= aligned_dot(
                gradients.grad_tx,
                gradients.grad_ty,
                gradients.grad_tz,
                gradients.grad_sx,
                gradients.grad_sy,
                gradients.grad_sz,
            ) > 0.0
        if self.config.require_gradT_gradRho_alignment:
            mask &= aligned_dot(
                gradients.grad_tx,
                gradients.grad_ty,
                gradients.grad_tz,
                gradients.grad_rx,
                gradients.grad_ry,
                gradients.grad_rz,
            ) > 0.0
        return mask

    def _compute_normals(self, gradients: GradientFields) -> NormalFields:
        """Compute unit shock normals from either temperature or pressure gradients."""

        if self.config.normal_field == "temperature":
            normal_x_raw = gradients.grad_tx
            normal_y_raw = gradients.grad_ty
            normal_z_raw = gradients.grad_tz
            normal_mag = gradients.grad_t_mag
        else:
            normal_x_raw = gradients.grad_px
            normal_y_raw = gradients.grad_py
            normal_z_raw = gradients.grad_pz
            normal_mag = gradients.grad_p_mag

        safe_mag = np.where(normal_mag > 0.0, normal_mag, 1.0)
        return NormalFields(
            normal_x=normal_x_raw / safe_mag,
            normal_y=normal_y_raw / safe_mag,
            normal_z=normal_z_raw / safe_mag,
        )

    def _compute_mach_fields(self, jumps: SampledJumps, gamma: float) -> MachFields:
        """Compute Mach estimates from pressure and temperature jumps."""

        shape = jumps.pressure_jump.shape
        mach_pressure = np.full(shape, np.nan, dtype=float)
        valid_pressure = jumps.pressure_jump > 1.0
        if np.any(valid_pressure):
            mach_pressure[valid_pressure] = np.sqrt(
                ((jumps.pressure_jump[valid_pressure] * (gamma + 1.0)) + (gamma - 1.0)) / (2.0 * gamma)
            )

        mach_temperature = np.full(shape, np.nan, dtype=float)
        valid_temperature = jumps.temperature_jump > 1.0
        if np.any(valid_temperature):
            values = [
                mach_from_temperature_jump(value, gamma, mach_max=self.config.mach_max)
                for value in jumps.temperature_jump[valid_temperature]
            ]
            mach_temperature[valid_temperature] = np.asarray(values, dtype=float)

        return MachFields(mach_pressure=mach_pressure, mach_temperature=mach_temperature)

    def _build_final_mask(
        self,
        shock_zone_mask: np.ndarray,
        jumps: SampledJumps,
        mach_fields: MachFields,
    ) -> np.ndarray:
        """Build the full filtered shock mask before optional center reduction."""

        jump_mask = np.ones(shock_zone_mask.shape, dtype=bool)
        if self.config.require_pressure_jump:
            jump_mask &= jumps.pressure_jump > 1.0
        if self.config.require_temperature_jump:
            jump_mask &= jumps.temperature_jump > 1.0
        if self.config.require_density_jump:
            jump_mask &= jumps.density_jump > 1.0

        mach_mask = np.nan_to_num(mach_fields.mach_pressure, nan=-np.inf) >= self.config.min_mach
        mach_mask |= np.nan_to_num(mach_fields.mach_temperature, nan=-np.inf) >= self.config.min_mach
        return shock_zone_mask & jump_mask & mach_mask

    def _reduce_to_centers(
        self,
        shock_mask: np.ndarray,
        div_v: np.ndarray,
        mach_fields: MachFields,
    ) -> np.ndarray:
        """Reduce connected shock regions to representative center cells."""

        labels, count = ndimage.label(shock_mask)
        if count == 0:
            return shock_mask

        reduced = np.zeros_like(shock_mask, dtype=bool)
        for label in range(1, count + 1):
            component = labels == label
            if not np.any(component):
                continue
            score = self._center_score(component, div_v, mach_fields)
            best_index = np.unravel_index(np.argmax(score), score.shape)
            reduced[best_index] = True
        return reduced

    def _center_score(self, component: np.ndarray, div_v: np.ndarray, mach_fields: MachFields) -> np.ndarray:
        """Return a component-local center score for representative cell selection."""

        masked_out = -np.inf
        compression = np.where(component, -div_v, masked_out)
        finite_pressure = np.where(component, np.nan_to_num(mach_fields.mach_pressure, nan=masked_out), masked_out)
        finite_temperature = np.where(
            component,
            np.nan_to_num(mach_fields.mach_temperature, nan=masked_out),
            masked_out,
        )

        if self.config.center_score == "compression":
            return compression
        if self.config.center_score == "mach_pressure":
            return finite_pressure
        if self.config.center_score == "mach_temperature":
            return finite_temperature

        finite_mach = np.maximum(finite_pressure, finite_temperature)
        finite_mach = np.where(np.isfinite(finite_mach), finite_mach, -1.0)
        return compression + finite_mach

    def _make_summary(
        self,
        cube: FluidCube,
        shock_zone_mask: np.ndarray,
        shock_mask: np.ndarray,
        full_shock_mask: np.ndarray,
        mach_fields: MachFields,
    ) -> dict[str, Any]:
        """Summarize both center-reduced and full shock populations."""

        total_cells = int(np.prod(cube.rho.shape))
        shock_cells = int(np.count_nonzero(shock_mask))
        full_shock_cells = int(np.count_nonzero(full_shock_mask))
        pmin, pmed, pmax = _finite_stats(mach_fields.mach_pressure[shock_mask])
        tmin, tmed, tmax = _finite_stats(mach_fields.mach_temperature[shock_mask])
        fpmin, fpmed, fpmax = _finite_stats(mach_fields.mach_pressure[full_shock_mask])
        ftmin, ftmed, ftmax = _finite_stats(mach_fields.mach_temperature[full_shock_mask])
        return {
            "grid_shape": cube.rho.shape,
            "total_cells": total_cells,
            "shock_zone_cells": int(np.count_nonzero(shock_zone_mask)),
            "shock_cells": shock_cells,
            "shock_fraction": shock_cells / total_cells,
            "full_shock_cells": full_shock_cells,
            "full_shock_fraction": full_shock_cells / total_cells,
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
            "min_mach": self.config.min_mach,
            "gamma": cube.gamma,
            "shock_width_cells": self.config.shock_width_cells,
            "reduce_to_centers": self.config.reduce_to_centers,
            "center_score": self.config.center_score,
            "sampling_method": self.config.sampling_method,
        }


def _finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    """Return finite min/median/max statistics, or NaN if none are finite."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))

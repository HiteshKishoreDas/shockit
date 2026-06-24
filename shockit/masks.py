"""Mask construction helpers for the shock finder."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import ShockFinderConfig
from .criteria import aligned_dot
from .derived import DerivedFields, GradientFields, MachFields
from .sampling import SampledJumps


def build_shock_zone_mask(
    derived: DerivedFields,
    gradients: GradientFields,
    config: ShockFinderConfig,
) -> np.ndarray:
    """Build the local shock-zone candidate mask."""

    mask = derived.div_v < config.div_v_threshold
    mask &= gradients.grad_t_mag > config.grad_T_min
    if config.require_gradT_gradS_alignment:
        mask &= aligned_dot(
            gradients.grad_tx,
            gradients.grad_ty,
            gradients.grad_tz,
            gradients.grad_sx,
            gradients.grad_sy,
            gradients.grad_sz,
        ) > 0.0
    if config.require_gradT_gradRho_alignment:
        mask &= aligned_dot(
            gradients.grad_tx,
            gradients.grad_ty,
            gradients.grad_tz,
            gradients.grad_rx,
            gradients.grad_ry,
            gradients.grad_rz,
        ) > 0.0
    return mask


def build_final_shock_mask(
    shock_zone_mask: np.ndarray,
    jumps: SampledJumps,
    mach_fields: MachFields,
    config: ShockFinderConfig,
) -> np.ndarray:
    """Build the unreduced shock mask after jump and Mach filtering."""

    jump_mask = np.ones(shock_zone_mask.shape, dtype=bool)
    if config.require_pressure_jump:
        jump_mask &= jumps.pressure_jump > 1.0
    if config.require_temperature_jump:
        jump_mask &= jumps.temperature_jump > 1.0
    if config.require_density_jump:
        jump_mask &= jumps.density_jump > 1.0

    mach_mask = np.nan_to_num(mach_fields.mach_pressure, nan=-np.inf) >= config.min_mach
    mach_mask |= np.nan_to_num(mach_fields.mach_temperature, nan=-np.inf) >= config.min_mach
    return shock_zone_mask & jump_mask & mach_mask


def reduce_to_centers(
    shock_mask: np.ndarray,
    div_v: np.ndarray,
    mach_fields: MachFields,
    center_score_mode: str,
) -> tuple[np.ndarray, int]:
    """Reduce each connected shock component to one representative center cell."""

    labels, count = ndimage.label(shock_mask)
    if count == 0:
        return np.zeros_like(shock_mask, dtype=bool), 0

    reduced = np.zeros_like(shock_mask, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        score = center_score(component, div_v, mach_fields, center_score_mode)
        best_flat_index = int(np.argmax(score))
        best_index = np.unravel_index(best_flat_index, score.shape)
        reduced[best_index] = True
    return reduced, count


def center_score(
    component: np.ndarray,
    div_v: np.ndarray,
    mach_fields: MachFields,
    mode: str,
) -> np.ndarray:
    """Return a component-local center score for representative cell selection."""

    masked_out = -np.inf
    compression = np.where(component, -div_v, masked_out)
    finite_pressure = np.where(component, np.nan_to_num(mach_fields.mach_pressure, nan=masked_out), masked_out)
    finite_temperature = np.where(component, np.nan_to_num(mach_fields.mach_temperature, nan=masked_out), masked_out)

    if mode == "compression":
        return compression
    if mode == "mach_pressure":
        return finite_pressure
    if mode == "mach_temperature":
        return finite_temperature

    combined_mach = np.maximum(finite_pressure, finite_temperature)
    compression_norm = _normalize_component_values(compression, component)
    mach_norm = _normalize_component_values(combined_mach, component)
    return np.where(component, compression_norm + mach_norm, masked_out)


def _normalize_component_values(values: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Normalize finite component values onto [0, 1], safely handling constants."""

    normalized = np.zeros(values.shape, dtype=float)
    finite_component = component & np.isfinite(values)
    if not np.any(finite_component):
        return normalized

    component_values = values[finite_component]
    vmin = float(np.min(component_values))
    vmax = float(np.max(component_values))
    if vmax <= vmin:
        normalized[finite_component] = 1.0
        return normalized

    normalized[finite_component] = (component_values - vmin) / (vmax - vmin)
    return normalized

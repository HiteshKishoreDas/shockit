"""Sampling helpers for upstream/downstream states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledJumps:
    """Jumps sampled across one-cell shock neighborhoods."""

    pressure_jump: np.ndarray
    temperature_jump: np.ndarray
    density_jump: np.ndarray
    pressure_up: np.ndarray
    pressure_down: np.ndarray


def dominant_axis(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Return dominant axis index per cell."""

    components = np.stack([np.abs(nx), np.abs(ny), np.abs(nz)], axis=0)
    return np.argmax(components, axis=0)


def axis_signs(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Return the sign along the chosen dominant axis."""

    signs = np.sign(nx)
    signs = np.where(axes == 1, np.sign(ny), signs)
    signs = np.where(axes == 2, np.sign(nz), signs)
    return np.where(signs == 0.0, 1.0, signs)


def sample_axis_offsets(field: np.ndarray, axes: np.ndarray, signs: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample plus/minus states along the nearest-axis normal direction."""

    plus = np.empty_like(field, dtype=float)
    minus = np.empty_like(field, dtype=float)
    for axis in range(3):
        axis_mask = axes == axis
        if not np.any(axis_mask):
            continue
        positive = axis_mask & (signs >= 0.0)
        negative = axis_mask & (signs < 0.0)
        if np.any(positive):
            plus_field = np.roll(field, -width, axis=axis)
            minus_field = np.roll(field, width, axis=axis)
            plus[positive] = plus_field[positive]
            minus[positive] = minus_field[positive]
        if np.any(negative):
            plus_field = np.roll(field, width, axis=axis)
            minus_field = np.roll(field, -width, axis=axis)
            plus[negative] = plus_field[negative]
            minus[negative] = minus_field[negative]
    return plus, minus


def sample_jumps(
    pressure: np.ndarray,
    temperature: np.ndarray,
    rho: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    width: int,
) -> SampledJumps:
    """Sample upstream/downstream jump ratios along the nearest normal axis."""

    axes = dominant_axis(nx, ny, nz)
    signs = axis_signs(nx, ny, nz, axes)

    pressure_plus, pressure_minus = sample_axis_offsets(pressure, axes, signs, width)
    temperature_plus, temperature_minus = sample_axis_offsets(temperature, axes, signs, width)
    rho_plus, rho_minus = sample_axis_offsets(rho, axes, signs, width)

    downstream_is_plus = pressure_plus >= pressure_minus
    pressure_down = np.where(downstream_is_plus, pressure_plus, pressure_minus)
    pressure_up = np.where(downstream_is_plus, pressure_minus, pressure_plus)
    temperature_down = np.where(downstream_is_plus, temperature_plus, temperature_minus)
    temperature_up = np.where(downstream_is_plus, temperature_minus, temperature_plus)
    rho_down = np.where(downstream_is_plus, rho_plus, rho_minus)
    rho_up = np.where(downstream_is_plus, rho_minus, rho_plus)

    return SampledJumps(
        pressure_jump=pressure_down / pressure_up,
        temperature_jump=temperature_down / temperature_up,
        density_jump=rho_down / rho_up,
        pressure_up=pressure_up,
        pressure_down=pressure_down,
    )

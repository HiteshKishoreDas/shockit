"""Sampling helpers for upstream/downstream states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


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
    method: str = "nearest_axis",
) -> SampledJumps:
    """Sample upstream/downstream jump ratios across a candidate shock."""

    if method == "nearest_axis":
        return _sample_jumps_nearest_axis(pressure, temperature, rho, nx, ny, nz, width)
    if method == "trilinear":
        return _sample_jumps_trilinear(pressure, temperature, rho, nx, ny, nz, width)
    raise ValueError(f"Unsupported sampling method: {method}")


def _sample_jumps_nearest_axis(
    pressure: np.ndarray,
    temperature: np.ndarray,
    rho: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    width: int,
) -> SampledJumps:
    """Sample jump ratios by stepping along the dominant grid axis."""

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


def _coordinate_grid(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        np.arange(shape[2], dtype=float),
        indexing="ij",
    )


def _sample_trilinear(field: np.ndarray, coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    stacked = np.vstack([axis.reshape(1, -1) for axis in coordinates])
    sampled = map_coordinates(field, stacked, order=1, mode="wrap")
    return sampled.reshape(field.shape)


def _sample_jumps_trilinear(
    pressure: np.ndarray,
    temperature: np.ndarray,
    rho: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    width: int,
) -> SampledJumps:
    """Sample jump ratios by trilinear interpolation along the local normal."""

    base_x, base_y, base_z = _coordinate_grid(pressure.shape)
    offset_x = width * nx
    offset_y = width * ny
    offset_z = width * nz

    plus_coordinates = (
        base_x + offset_x,
        base_y + offset_y,
        base_z + offset_z,
    )
    minus_coordinates = (
        base_x - offset_x,
        base_y - offset_y,
        base_z - offset_z,
    )

    pressure_plus = _sample_trilinear(pressure, plus_coordinates)
    pressure_minus = _sample_trilinear(pressure, minus_coordinates)
    temperature_plus = _sample_trilinear(temperature, plus_coordinates)
    temperature_minus = _sample_trilinear(temperature, minus_coordinates)
    rho_plus = _sample_trilinear(rho, plus_coordinates)
    rho_minus = _sample_trilinear(rho, minus_coordinates)

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

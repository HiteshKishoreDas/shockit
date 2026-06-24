"""Rankine-Hugoniot jump relations and inversion utilities."""

from __future__ import annotations

import math


def pressure_jump_from_mach(mach: float, gamma: float) -> float:
    """Return the ideal-gas pressure jump for a normal shock."""

    m2 = mach * mach
    return (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)


def density_jump_from_mach(mach: float, gamma: float) -> float:
    """Return the ideal-gas density jump for a normal shock."""

    m2 = mach * mach
    return ((gamma + 1.0) * m2) / ((gamma - 1.0) * m2 + 2.0)


def temperature_jump_from_mach(mach: float, gamma: float) -> float:
    """Return the ideal-gas temperature jump for a normal shock."""

    return pressure_jump_from_mach(mach, gamma) / density_jump_from_mach(mach, gamma)


def mach_from_pressure_jump(pressure_jump: float, gamma: float) -> float:
    """Invert a pressure jump to upstream Mach number."""

    if pressure_jump <= 1.0:
        return math.nan
    m2 = ((pressure_jump * (gamma + 1.0)) + (gamma - 1.0)) / (2.0 * gamma)
    return math.sqrt(m2)


def mach_from_temperature_jump(
    temperature_jump: float,
    gamma: float,
    mach_max: float = 1000.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Invert a temperature jump to upstream Mach number via bisection."""

    if temperature_jump <= 1.0:
        return math.nan

    low = 1.0
    high = mach_max
    low_value = temperature_jump_from_mach(low, gamma) - temperature_jump
    high_value = temperature_jump_from_mach(high, gamma) - temperature_jump
    if low_value > 0.0 or high_value < 0.0:
        return math.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_value = temperature_jump_from_mach(mid, gamma) - temperature_jump
        if abs(mid_value) < tol:
            return mid
        if mid_value > 0.0:
            high = mid
        else:
            low = mid
    return math.nan

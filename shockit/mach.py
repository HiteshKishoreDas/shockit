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
) -> float:
    """Invert a temperature jump to upstream Mach number analytically."""

    if temperature_jump <= 1.0:
        return math.nan

    gamma_plus_one = gamma + 1.0
    gamma_minus_one = gamma - 1.0
    coefficient_a = 2.0 * gamma * gamma_minus_one
    coefficient_b = (
        -(gamma * gamma)
        + (6.0 * gamma)
        - 1.0
        - (temperature_jump * gamma_plus_one * gamma_plus_one)
    )
    coefficient_c = -2.0 * gamma_minus_one
    discriminant = (coefficient_b * coefficient_b) - (4.0 * coefficient_a * coefficient_c)
    if discriminant < 0.0:
        return math.nan

    mach_squared = (-coefficient_b + math.sqrt(discriminant)) / (2.0 * coefficient_a)
    if mach_squared <= 1.0:
        return math.nan

    mach = math.sqrt(mach_squared)
    if mach > mach_max:
        return math.nan
    return mach

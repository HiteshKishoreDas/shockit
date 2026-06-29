"""Result container for shock finding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import ChunkedFieldReference


@dataclass
class ShockFinderResult:
    """Results returned by ShockFinder.

    `shock_zone_mask` marks local compression and gradient-alignment candidates.
    `full_shock_mask` marks unreduced shock cells that also pass jump and Mach
    filtering.
    `shock_mask` is the final public mask and may be center-reduced.
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
class ChunkedShockFinderResult:
    """Chunked on-disk result with field handles parallel to `ShockFinderResult`."""

    output_root: Path
    shock_mask: ChunkedFieldReference
    shock_zone_mask: ChunkedFieldReference
    mach_temperature: ChunkedFieldReference
    mach_pressure: ChunkedFieldReference
    compression: ChunkedFieldReference
    div_v: ChunkedFieldReference
    temperature: ChunkedFieldReference
    entropy: ChunkedFieldReference
    temperature_jump: ChunkedFieldReference
    pressure_jump: ChunkedFieldReference
    density_jump: ChunkedFieldReference
    normal_x: ChunkedFieldReference
    normal_y: ChunkedFieldReference
    normal_z: ChunkedFieldReference
    summary: dict[str, Any]
    full_shock_mask: ChunkedFieldReference | None = None

"""Data model definitions for uniform fluid cubes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FluidCube:
    """Uniform 3D cube of primitive ideal-gas hydrodynamic fields."""

    rho: np.ndarray
    pressure: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    gamma: float = 5.0 / 3.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = {
            "rho": np.asarray(self.rho, dtype=float),
            "pressure": np.asarray(self.pressure, dtype=float),
            "vx": np.asarray(self.vx, dtype=float),
            "vy": np.asarray(self.vy, dtype=float),
            "vz": np.asarray(self.vz, dtype=float),
        }
        shape = arrays["rho"].shape
        if len(shape) != 3:
            raise ValueError("FluidCube fields must be 3D arrays.")
        for name, value in arrays.items():
            if value.shape != shape:
                raise ValueError(f"Field {name} has shape {value.shape}, expected {shape}.")
            setattr(self, name, value)
        if np.any(self.rho <= 0.0):
            raise ValueError("rho must be strictly positive.")
        if np.any(self.pressure <= 0.0):
            raise ValueError("pressure must be strictly positive.")
        if self.dx <= 0.0 or self.dy <= 0.0 or self.dz <= 0.0:
            raise ValueError("dx, dy, dz must be strictly positive.")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1.")

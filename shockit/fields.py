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
        """Validate primitive fields and grid metadata for a uniform cube."""

        arrays = {
            "rho": _as_float_array("rho", self.rho),
            "pressure": _as_float_array("pressure", self.pressure),
            "vx": _as_float_array("vx", self.vx),
            "vy": _as_float_array("vy", self.vy),
            "vz": _as_float_array("vz", self.vz),
        }
        shape = arrays["rho"].shape
        if len(shape) != 3:
            raise ValueError("FluidCube fields must be 3D arrays.")
        for name, value in arrays.items():
            if value.shape != shape:
                raise ValueError(f"Field {name} has shape {value.shape}, expected {shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Field {name} must contain only finite values.")
            setattr(self, name, value)
        if np.any(self.rho <= 0.0):
            raise ValueError("rho must be strictly positive.")
        if np.any(self.pressure <= 0.0):
            raise ValueError("pressure must be strictly positive.")
        if not np.isfinite(self.dx) or not np.isfinite(self.dy) or not np.isfinite(self.dz):
            raise ValueError("dx, dy, dz must be finite.")
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite.")
        if self.dx <= 0.0 or self.dy <= 0.0 or self.dz <= 0.0:
            raise ValueError("dx, dy, dz must be strictly positive.")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1.")


def _as_float_array(name: str, value: Any) -> np.ndarray:
    """Convert one primitive field to a finite float array with useful errors."""

    if isinstance(value, np.lib.npyio.NpzFile):
        keys = ", ".join(sorted(value.files))
        raise ValueError(
            f"Field {name} received a NumPy .npz archive, not an array. "
            f"Load one dataset from the archive first, for example np.load(...)[\"arr_0\"]. "
            f"Available keys: {keys}."
        )
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {name} could not be converted to a float array.") from exc

"""Helpers for candidate shock criteria."""

from __future__ import annotations

import numpy as np


def aligned_dot(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> np.ndarray:
    """Return vector dot product."""

    return ax * bx + ay * by + az * bz

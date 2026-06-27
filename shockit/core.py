"""Compatibility module for core concepts."""

from .chunked import join_all_fields, join_chunked_field, save_chunked_field
from .chunked_center_reduction import reduce_chunked_shock_outputs
from .chunked_finder import run_chunked_npz_shock_finder
from .config import ShockFinderConfig
from .fields import FluidCube
from .finder import ShockFinder
from .npz_pipeline import detect_npz_layout, run_npz_shock_finder
from .result import ShockFinderResult

__all__ = [
    "FluidCube",
    "ShockFinder",
    "ShockFinderConfig",
    "ShockFinderResult",
    "detect_npz_layout",
    "join_all_fields",
    "join_chunked_field",
    "reduce_chunked_shock_outputs",
    "run_npz_shock_finder",
    "run_chunked_npz_shock_finder",
    "save_chunked_field",
]

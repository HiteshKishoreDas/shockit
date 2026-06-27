"""Public package exports for shockit."""

from importlib.metadata import PackageNotFoundError, version

from .chunked import join_all_fields, join_chunked_field, save_chunked_field
from .chunked_center_reduction import reduce_chunked_shock_outputs
from .chunked_finder import run_chunked_npz_shock_finder
from .config import ShockFinderConfig
from .fields import FluidCube
from .finder import ShockFinder
from .npz_pipeline import detect_npz_layout, run_npz_shock_finder
from .io_hdf5 import load_fluid_cube_from_hdf5
from .output import save_result_hdf5
from .result import ShockFinderResult

try:
    __version__ = version("shockit")
except PackageNotFoundError:  # pragma: no cover - fallback for editable local imports
    __version__ = "0+unknown"

__all__ = [
    "FluidCube",
    "ShockFinder",
    "ShockFinderConfig",
    "ShockFinderResult",
    "detect_npz_layout",
    "join_all_fields",
    "join_chunked_field",
    "reduce_chunked_shock_outputs",
    "load_fluid_cube_from_hdf5",
    "run_npz_shock_finder",
    "run_chunked_npz_shock_finder",
    "save_chunked_field",
    "save_result_hdf5",
    "__version__",
]

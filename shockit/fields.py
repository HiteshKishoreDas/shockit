"""Data model definitions for uniform fluid cubes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .chunking.layout import ChunkLayout, load_chunk_layout

FieldSource = Any


@dataclass(frozen=True)
class ChunkedFieldReference:
    """Lightweight reference to one chunked field directory."""

    path: Path
    layout: ChunkLayout

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.shape

    def load(self) -> np.ndarray:
        """Reconstruct the full field array in memory."""

        with np.load(self.path / self.layout.specs[0].path.name) as first_chunk:
            full_array = np.empty(self.layout.shape, dtype=first_chunk["values"].dtype)
        for spec, values in self.iter_chunks():
            chunk_slice = tuple(slice(spec.start[axis], spec.stop[axis]) for axis in range(3))
            full_array[chunk_slice] = values
        return full_array

    def iter_chunks(self):
        """Yield `(ChunkSpec, values)` pairs without reconstructing the full field."""

        for spec in self.layout.specs:
            with np.load(self.path / spec.path.name) as chunk:
                yield spec, np.asarray(chunk["values"])


@dataclass
class FluidCube:
    """Uniform 3D cube of primitive ideal-gas hydrodynamic fields."""

    rho: FieldSource
    pressure: FieldSource
    vx: FieldSource
    vy: FieldSource
    vz: FieldSource
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    gamma: float = 5.0 / 3.0
    metadata: dict[str, Any] = field(default_factory=dict)
    storage_mode: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate primitive fields and grid metadata for a uniform cube."""

        field_values = {
            "rho": self.rho,
            "pressure": self.pressure,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
        }
        pathlike_fields = {
            name: value
            for name, value in field_values.items()
            if _is_chunked_field_source(value)
        }
        if pathlike_fields and len(pathlike_fields) != len(field_values):
            raise ValueError(
                "FluidCube fields must be either all in-memory arrays or all chunked field paths."
            )
        if len(pathlike_fields) == len(field_values):
            self._validate_chunked_fields(field_values)
            return
        self._validate_in_memory_fields(field_values)

    @property
    def is_in_memory(self) -> bool:
        return self.storage_mode == "in_memory"

    @property
    def is_chunked(self) -> bool:
        return self.storage_mode == "chunked"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rho.shape

    def _validate_in_memory_fields(self, field_values: dict[str, FieldSource]) -> None:
        arrays = {
            name: _as_float_array(name, value)
            for name, value in field_values.items()
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
        self.storage_mode = "in_memory"

    def _validate_chunked_fields(self, field_values: dict[str, FieldSource]) -> None:
        refs = {
            name: _as_chunked_field_reference(name, value)
            for name, value in field_values.items()
        }
        base_signature: tuple[
            tuple[int, int, int],
            tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
        ] | None = None
        for name, ref in refs.items():
            signature = _layout_signature(ref.layout)
            if base_signature is None:
                base_signature = signature
            elif signature != base_signature:
                raise ValueError(
                    f"Chunked field {name} has a layout incompatible with the other primitive fields."
                )
            setattr(self, name, ref)
        if not np.isfinite(self.dx) or not np.isfinite(self.dy) or not np.isfinite(self.dz):
            raise ValueError("dx, dy, dz must be finite.")
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite.")
        if self.dx <= 0.0 or self.dy <= 0.0 or self.dz <= 0.0:
            raise ValueError("dx, dy, dz must be strictly positive.")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1.")
        self.storage_mode = "chunked"


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


def _is_chunked_field_source(value: Any) -> bool:
    return isinstance(value, (str, Path, ChunkedFieldReference))


def _as_chunked_field_reference(name: str, value: Any) -> ChunkedFieldReference:
    if isinstance(value, ChunkedFieldReference):
        return value
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"Field {name} must be a path-like chunk directory when using chunked FluidCube storage."
        )
    field_dir = Path(value)
    if not field_dir.exists():
        raise FileNotFoundError(f"Chunked field directory for {name} does not exist: {field_dir}")
    return ChunkedFieldReference(path=field_dir, layout=load_chunk_layout(field_dir))


def _layout_signature(
    layout: ChunkLayout,
) -> tuple[tuple[int, int, int], tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]]:
    return (
        layout.shape,
        tuple((spec.start, spec.stop) for spec in layout.specs),
    )

"""Chunked field writers and reconstruction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .layout import ChunkLayout, ChunkSpec, load_chunk_layout
from .reader import ChunkedFieldReader, NpzFieldReader


class ChunkedFieldWriter(Protocol):
    """Abstract chunked field writer."""

    def write_core(self, field_name: str, spec: ChunkSpec, values: np.ndarray) -> None: ...


class ChunkedOutputStore(Protocol):
    """Abstract chunked output store."""

    @property
    def root(self) -> Path: ...

    @property
    def layout(self) -> ChunkLayout: ...

    def field_reader(self, field_name: str) -> ChunkedFieldReader: ...

    def write_core(self, field_name: str, spec: ChunkSpec, values: np.ndarray) -> None: ...

    def write_summary(self, summary: dict[str, Any]) -> None: ...

    def read_summary(self) -> dict[str, Any]: ...


class NpzChunkedOutput:
    """Chunked `.npz` output store."""

    def __init__(self, root: str | Path, layout: ChunkLayout) -> None:
        self.root = Path(root)
        self.layout = layout

    def write_core(self, field_name: str, spec: ChunkSpec, values: np.ndarray) -> None:
        field_dir = self.root / field_name
        field_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            field_dir / spec.path.name,
            values=values,
            start=np.array(spec.start, dtype=np.int64),
            stop=np.array(spec.stop, dtype=np.int64),
            full_shape=np.array(spec.full_shape, dtype=np.int64),
        )

    def field_reader(self, field_name: str) -> NpzFieldReader:
        field_dir = self.root / field_name
        first_chunk = self.layout.specs[0].path.name
        with np.load(field_dir / first_chunk) as chunk:
            dtype = chunk["values"].dtype
        return NpzFieldReader(field_dir=field_dir, layout=self.layout, dtype=dtype)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    def read_summary(self) -> dict[str, Any]:
        summary_path = self.root / "summary.json"
        if not summary_path.exists():
            return {}
        return json.loads(summary_path.read_text())


def join_chunked_field(field_dir: str | Path) -> np.ndarray:
    """Reconstruct a full array from chunked field files."""

    layout = load_chunk_layout(field_dir)
    with np.load(layout.specs[0].path) as first_chunk:
        full_array = np.empty(layout.shape, dtype=first_chunk["values"].dtype)

    for spec in layout.specs:
        with np.load(spec.path) as chunk:
            chunk_slice = tuple(slice(spec.start[axis], spec.stop[axis]) for axis in range(3))
            full_array[chunk_slice] = chunk["values"]
    return full_array


def join_all_fields(root_dir: str | Path) -> dict[str, np.ndarray]:
    """Reconstruct all chunked fields in one root directory."""

    root_path = Path(root_dir)
    field_arrays: dict[str, np.ndarray] = {}
    for field_dir in sorted(path for path in root_path.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if not list(field_dir.glob("chunk_*.npz")):
            continue
        field_arrays[field_dir.name] = join_chunked_field(field_dir)
    return field_arrays

"""Compatibility wrapper for the explicit `shockit.chunking` package."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunking.layout import ChunkShape, ChunkSpec, balanced_axis_slices, iter_balanced_slices, load_chunk_layout, save_chunked_field
from .chunking.writer import join_all_fields, join_chunked_field


def write_chunk_file(field_dir: str | Path, spec: ChunkSpec, values: np.ndarray) -> None:
    """Write one chunk file using an existing core-grid spec."""

    output_dir = Path(field_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / spec.path.name,
        values=values,
        start=np.array(spec.start, dtype=np.int64),
        stop=np.array(spec.stop, dtype=np.int64),
        full_shape=np.array(spec.full_shape, dtype=np.int64),
    )


def load_chunk_specs(field_dir: str | Path) -> list[ChunkSpec]:
    """Compatibility wrapper returning chunk specs from one field directory."""

    return list(load_chunk_layout(field_dir).specs)

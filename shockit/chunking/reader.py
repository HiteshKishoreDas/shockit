"""Chunked field readers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np

from .layout import ChunkLayout, ChunkSpec, load_chunk_layout

DEFAULT_FIELD_MAPPING = {
    "rho": "rho",
    "pressure": "prs",
    "vx": "v1",
    "vy": "v2",
    "vz": "v3",
}


class ChunkedFieldReader(Protocol):
    """Abstract chunked field reader."""

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def layout(self) -> ChunkLayout: ...

    def read_core(self, spec: ChunkSpec) -> np.ndarray: ...

    def read_with_halo(self, spec: ChunkSpec, halo: int) -> np.ndarray: ...


class ChunkedInputStore(Protocol):
    """Abstract chunked input store."""

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def layout(self) -> ChunkLayout: ...

    def field_reader(self, field_name: str) -> ChunkedFieldReader: ...


@dataclass(frozen=True)
class NpzFieldReader:
    """Read one field from a chunked `.npz` directory."""

    field_dir: Path
    layout: ChunkLayout
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.shape

    def read_core(self, spec: ChunkSpec) -> np.ndarray:
        with np.load(self.field_dir / spec.path.name) as chunk:
            return np.asarray(chunk["values"])

    def read_with_halo(self, spec: ChunkSpec, halo: int) -> np.ndarray:
        request_slices = tuple(
            slice(spec.start[axis] - halo, spec.stop[axis] + halo)
            for axis in range(3)
        )
        output = np.empty(
            tuple(request.stop - request.start for request in request_slices),
            dtype=self.dtype,
        )
        axis_segments = [
            _periodic_axis_segments(self.shape[axis], request.start, request.stop)
            for axis, request in enumerate(request_slices)
        ]
        for x_segment, y_segment, z_segment in product(*axis_segments):
            source_slices = (
                slice(x_segment[0], x_segment[1]),
                slice(y_segment[0], y_segment[1]),
                slice(z_segment[0], z_segment[1]),
            )
            dest_slices = (
                slice(x_segment[2], x_segment[3]),
                slice(y_segment[2], y_segment[3]),
                slice(z_segment[2], z_segment[3]),
            )
            _fill_region_from_specs(
                output=output,
                dest_slices=dest_slices,
                source_slices=source_slices,
                field_dir=self.field_dir,
                specs=self.layout.specs,
            )
        return output


class NpzChunkedInput:
    """Chunked `.npz` input store."""

    def __init__(
        self,
        root: str | Path,
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.field_mapping = dict(DEFAULT_FIELD_MAPPING if field_mapping is None else field_mapping)
        self._layout: ChunkLayout | None = None
        self._readers: dict[str, NpzFieldReader] = {}
        self._build_readers()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.shape

    @property
    def layout(self) -> ChunkLayout:
        assert self._layout is not None
        return self._layout

    def field_reader(self, field_name: str) -> NpzFieldReader:
        return self._readers[field_name]

    def _build_readers(self) -> None:
        base_signature: tuple[tuple[int, int, int], tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] | None = None
        for logical_name, directory_name in self.field_mapping.items():
            field_dir = self.root / directory_name
            layout = load_chunk_layout(field_dir)
            signature = _layout_signature(layout)
            if base_signature is None:
                base_signature = signature
                self._layout = layout
            elif signature != base_signature:
                raise ValueError(f"Chunk layout mismatch in {field_dir}.")
            with np.load(layout.specs[0].path) as first_chunk:
                dtype = first_chunk["values"].dtype
            self._readers[logical_name] = NpzFieldReader(field_dir=field_dir, layout=layout, dtype=dtype)


def _layout_signature(
    layout: ChunkLayout,
) -> tuple[tuple[int, int, int], tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]]:
    return (
        layout.shape,
        tuple((spec.start, spec.stop) for spec in layout.specs),
    )


def _periodic_axis_segments(axis_size: int, start: int, stop: int) -> list[tuple[int, int, int, int]]:
    total = stop - start
    segments: list[tuple[int, int, int, int]] = []
    local_start = 0
    global_start = start
    while local_start < total:
        wrapped_start = global_start % axis_size
        available = axis_size - wrapped_start
        segment_length = min(available, total - local_start)
        segments.append(
            (
                wrapped_start,
                wrapped_start + segment_length,
                local_start,
                local_start + segment_length,
            )
        )
        local_start += segment_length
        global_start += segment_length
    return segments


def _fill_region_from_specs(
    output: np.ndarray,
    dest_slices: tuple[slice, slice, slice],
    source_slices: tuple[slice, slice, slice],
    field_dir: Path,
    specs: tuple[ChunkSpec, ...],
) -> None:
    for spec in specs:
        overlap = _intersection_slices(source_slices, spec)
        if overlap is None:
            continue
        source_overlap, chunk_overlap = overlap
        dest_overlap = tuple(
            slice(
                dest_slices[axis].start + (source_overlap[axis].start - source_slices[axis].start),
                dest_slices[axis].start + (source_overlap[axis].stop - source_slices[axis].start),
            )
            for axis in range(3)
        )
        with np.load(field_dir / spec.path.name) as chunk:
            output[dest_overlap] = chunk["values"][chunk_overlap]


def _intersection_slices(
    source_slices: tuple[slice, slice, slice],
    spec: ChunkSpec,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    source_overlap: list[slice] = []
    chunk_overlap: list[slice] = []
    for axis in range(3):
        start = max(source_slices[axis].start, spec.start[axis])
        stop = min(source_slices[axis].stop, spec.stop[axis])
        if start >= stop:
            return None
        source_overlap.append(slice(start, stop))
        chunk_overlap.append(slice(start - spec.start[axis], stop - spec.start[axis]))
    return tuple(source_overlap), tuple(chunk_overlap)

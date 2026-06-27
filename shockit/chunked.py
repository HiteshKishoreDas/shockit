"""Chunked NumPy cube helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChunkSpec:
    """One chunk file and its core-grid extent."""

    path: Path
    start: tuple[int, int, int]
    stop: tuple[int, int, int]
    full_shape: tuple[int, int, int]


ChunkShape = int | tuple[int, int, int]


def balanced_axis_slices(axis_size: int, target_chunk_size: int) -> list[slice]:
    """Split one axis into nearly equal slices with size close to target."""

    chunk_count = max(1, math.ceil(axis_size / target_chunk_size))
    boundaries = np.linspace(0, axis_size, num=chunk_count + 1, dtype=int)
    return [
        slice(int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(chunk_count)
    ]


def iter_balanced_slices(shape: tuple[int, int, int], target_chunk_size: ChunkShape):
    """Yield balanced 3D chunk slices for a full cube shape."""

    axis_chunk_sizes = _normalize_chunk_shape(target_chunk_size)
    axis_slices = [
        balanced_axis_slices(axis_size, axis_chunk_sizes[axis]) for axis, axis_size in enumerate(shape)
    ]
    for x_slice, y_slice, z_slice in product(*axis_slices):
        yield x_slice, y_slice, z_slice


def _normalize_chunk_shape(chunk_size: ChunkShape) -> tuple[int, int, int]:
    if isinstance(chunk_size, int):
        return (chunk_size, chunk_size, chunk_size)
    return chunk_size


def save_chunked_field(
    field_name: str,
    values: np.ndarray,
    output_dir: str | Path,
    target_chunk_size: int,
) -> None:
    """Write one 3D field as a directory of chunked .npz files."""

    output_path = Path(output_dir)
    field_dir = output_path / field_name
    field_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk_slice in enumerate(
        iter_balanced_slices(values.shape, target_chunk_size)
    ):
        chunk_values = values[chunk_slice]
        chunk_path = field_dir / (
            f"chunk_{chunk_index:04d}"
            f"_x{chunk_slice[0].start:04d}-{chunk_slice[0].stop:04d}"
            f"_y{chunk_slice[1].start:04d}-{chunk_slice[1].stop:04d}"
            f"_z{chunk_slice[2].start:04d}-{chunk_slice[2].stop:04d}.npz"
        )
        np.savez(
            chunk_path,
            values=chunk_values,
            start=np.array(
                [chunk_slice[0].start, chunk_slice[1].start, chunk_slice[2].start],
                dtype=np.int64,
            ),
            stop=np.array(
                [chunk_slice[0].stop, chunk_slice[1].stop, chunk_slice[2].stop],
                dtype=np.int64,
            ),
            full_shape=np.array(values.shape, dtype=np.int64),
        )


def write_chunk_file(field_dir: str | Path, spec: ChunkSpec, values: np.ndarray) -> None:
    """Write one chunk file using an existing core-grid spec."""

    output_dir = Path(field_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / spec.path.name
    np.savez(
        chunk_path,
        values=values,
        start=np.array(spec.start, dtype=np.int64),
        stop=np.array(spec.stop, dtype=np.int64),
        full_shape=np.array(spec.full_shape, dtype=np.int64),
    )


def load_chunk_specs(field_dir: str | Path) -> list[ChunkSpec]:
    """Read chunk metadata from one field directory."""

    field_path = Path(field_dir)
    chunk_files = sorted(field_path.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {field_path}.")

    specs: list[ChunkSpec] = []
    for chunk_path in chunk_files:
        with np.load(chunk_path) as chunk:
            specs.append(
                ChunkSpec(
                    path=chunk_path,
                    start=tuple(int(value) for value in chunk["start"]),
                    stop=tuple(int(value) for value in chunk["stop"]),
                    full_shape=tuple(int(value) for value in chunk["full_shape"]),
                )
            )
    return specs


def join_chunked_field(field_dir: str | Path) -> np.ndarray:
    """Reconstruct a full array from chunked field files."""

    specs = load_chunk_specs(field_dir)
    with np.load(specs[0].path) as first_chunk:
        full_array = np.empty(specs[0].full_shape, dtype=first_chunk["values"].dtype)

    for spec in specs:
        with np.load(spec.path) as chunk:
            chunk_slice = tuple(slice(spec.start[axis], spec.stop[axis]) for axis in range(3))
            full_array[chunk_slice] = chunk["values"]
    return full_array


def join_all_fields(root_dir: str | Path) -> dict[str, np.ndarray]:
    """Reconstruct all chunked fields in one root directory."""

    root_path = Path(root_dir)
    field_arrays: dict[str, np.ndarray] = {}
    for field_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        field_arrays[field_dir.name] = join_chunked_field(field_dir)
    return field_arrays

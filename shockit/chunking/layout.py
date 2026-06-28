"""Chunk layout primitives for streamed chunked workflows."""

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
    grid_index: tuple[int, int, int] = (0, 0, 0)

    @property
    def core_shape(self) -> tuple[int, int, int]:
        return tuple(self.stop[axis] - self.start[axis] for axis in range(3))


ChunkShape = int | tuple[int, int, int]


@dataclass(frozen=True)
class ChunkLayout:
    """Resolved chunk grid shared by one field store."""

    specs: tuple[ChunkSpec, ...]
    axis_slices: tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.specs[0].full_shape

    def spec_for_grid_index(self, grid_index: tuple[int, int, int]) -> ChunkSpec:
        for spec in self.specs:
            if spec.grid_index == grid_index:
                return spec
        raise KeyError(f"Missing chunk at grid index {grid_index}.")

    def iter_face_neighbor_pairs(self, periodic: bool = True):
        """Yield chunk pairs that share a face in the positive axis direction."""

        for spec in self.specs:
            for axis, axis_chunks in enumerate(self.axis_slices):
                if len(axis_chunks) == 1 and not periodic:
                    continue
                neighbor_index = list(spec.grid_index)
                if spec.grid_index[axis] + 1 < len(axis_chunks):
                    neighbor_index[axis] += 1
                elif periodic:
                    neighbor_index[axis] = 0
                else:
                    continue
                yield axis, spec, self.spec_for_grid_index(tuple(neighbor_index))


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

    axis_chunk_sizes = normalize_chunk_shape(target_chunk_size)
    axis_slices = [
        balanced_axis_slices(axis_size, axis_chunk_sizes[axis])
        for axis, axis_size in enumerate(shape)
    ]
    for x_slice, y_slice, z_slice in product(*axis_slices):
        yield x_slice, y_slice, z_slice


def normalize_chunk_shape(chunk_size: ChunkShape) -> tuple[int, int, int]:
    """Return a per-axis chunk shape."""

    if isinstance(chunk_size, int):
        return (chunk_size, chunk_size, chunk_size)
    return chunk_size


def save_chunked_field(
    field_name: str,
    values: np.ndarray,
    output_dir: str | Path,
    target_chunk_size: ChunkShape,
) -> None:
    """Write one 3D field as a directory of chunked `.npz` files."""

    output_path = Path(output_dir)
    field_dir = output_path / field_name
    field_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk_slice in enumerate(iter_balanced_slices(values.shape, target_chunk_size)):
        np.savez(
            field_dir / _chunk_filename(chunk_index, chunk_slice),
            values=values[chunk_slice],
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


def load_chunk_layout(field_dir: str | Path) -> ChunkLayout:
    """Read chunk metadata from one field directory."""

    field_path = Path(field_dir)
    chunk_files = sorted(field_path.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {field_path}.")

    raw_specs: list[tuple[Path, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
    for chunk_path in chunk_files:
        with np.load(chunk_path) as chunk:
            raw_specs.append(
                (
                    chunk_path,
                    tuple(int(value) for value in chunk["start"]),
                    tuple(int(value) for value in chunk["stop"]),
                    tuple(int(value) for value in chunk["full_shape"]),
                )
            )

    full_shapes = {full_shape for _, _, _, full_shape in raw_specs}
    if len(full_shapes) != 1:
        raise ValueError(f"Inconsistent full shapes in {field_path}.")

    axis_bounds = tuple(
        tuple(
            slice(start, stop)
            for start, stop in sorted(
                {(spec[1][axis], spec[2][axis]) for spec in raw_specs},
                key=lambda pair: pair[0],
            )
        )
        for axis in range(3)
    )
    axis_lookup = [
        {(axis_slice.start, axis_slice.stop): index for index, axis_slice in enumerate(axis_slices)}
        for axis_slices in axis_bounds
    ]
    specs = [
        ChunkSpec(
            path=path,
            start=start,
            stop=stop,
            full_shape=full_shape,
            grid_index=tuple(axis_lookup[axis][(start[axis], stop[axis])] for axis in range(3)),
        )
        for path, start, stop, full_shape in raw_specs
    ]
    specs.sort(key=lambda spec: spec.grid_index)

    expected_chunk_count = math.prod(len(axis_slices) for axis_slices in axis_bounds)
    if len(specs) != expected_chunk_count:
        raise ValueError(f"Chunk coverage in {field_path} does not form a complete regular grid.")
    if len({spec.grid_index for spec in specs}) != len(specs):
        raise ValueError(f"Chunk coverage in {field_path} has duplicate grid indices.")

    return ChunkLayout(specs=tuple(specs), axis_slices=axis_bounds)


def _chunk_filename(chunk_index: int, chunk_slice: tuple[slice, slice, slice]) -> str:
    return (
        f"chunk_{chunk_index:04d}"
        f"_x{chunk_slice[0].start:04d}-{chunk_slice[0].stop:04d}"
        f"_y{chunk_slice[1].start:04d}-{chunk_slice[1].stop:04d}"
        f"_z{chunk_slice[2].start:04d}-{chunk_slice[2].stop:04d}.npz"
    )

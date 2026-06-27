"""Chunk-aware shock-finder pipeline for chunked .npz cube directories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .chunked import ChunkSpec, load_chunk_specs, write_chunk_file
from .chunked_center_reduction import reduce_chunked_shock_outputs
from .config import ShockFinderConfig
from .fields import FluidCube
from .finder import analyze_cube_pass

ProgressCallback = Callable[[str], None]
ProgressReporter = ProgressCallback | Literal[True, False] | None

INPUT_FIELD_DIRS = {
    "rho": "rho",
    "pressure": "prs",
    "vx": "v1",
    "vy": "v2",
    "vz": "v3",
}

OUTPUT_FIELDS = (
    "shock_mask",
    "shock_zone_mask",
    "full_shock_mask",
    "mach_temperature",
    "mach_pressure",
    "compression",
    "div_v",
    "temperature",
    "entropy",
    "temperature_jump",
    "pressure_jump",
    "density_jump",
    "normal_x",
    "normal_y",
    "normal_z",
)


@dataclass
class _FieldReader:
    field_dir: Path
    specs: list[ChunkSpec]
    dtype: np.dtype

    def read_region(self, request_slices: tuple[slice, slice, slice]) -> np.ndarray:
        shape = self.specs[0].full_shape
        output_shape = tuple(
            request_slices[axis].stop - request_slices[axis].start for axis in range(3)
        )
        output = np.empty(output_shape, dtype=self.dtype)
        axis_segments = [
            _periodic_axis_segments(shape[axis], request_slices[axis].start, request_slices[axis].stop)
            for axis in range(3)
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
            _fill_region_from_specs(output, dest_slices, source_slices, self.specs)
        return output


def run_chunked_npz_shock_finder(
    input_root: str | Path,
    output_root: str | Path,
    config: ShockFinderConfig,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    gamma: float = 5.0 / 3.0,
    progress: ProgressReporter = True,
) -> dict[str, object]:
    """Run the shock finder directly on chunked input cubes and emit chunked outputs."""

    progress_callback: ProgressCallback | None
    if progress is True or progress is None:
        progress_callback = print
    elif progress is False:
        progress_callback = None
    else:
        progress_callback = progress

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    input_path = Path(input_root)
    output_path = Path(output_root)
    readers = _build_field_readers(input_path)
    base_specs = readers["rho"].specs
    shape = base_specs[0].full_shape
    halo = max(1, config.shock_width_cells)

    report(f"[1/2] Processing {len(base_specs)} chunk files with halo width {halo}")
    shock_zone_cells = 0
    full_shock_cells = 0
    mach_pressure_values: list[np.ndarray] = []
    mach_temperature_values: list[np.ndarray] = []

    for chunk_index, spec in enumerate(base_specs, start=1):
        report(f"[1/2] Chunk {chunk_index}/{len(base_specs)} {spec.start}->{spec.stop}")
        local_cube = _read_local_cube(
            readers,
            spec,
            halo,
            dx=dx,
            dy=dy,
            dz=dz,
            gamma=gamma,
        )
        local_pass = analyze_cube_pass(local_cube, config)
        core_crop = tuple(slice(halo, halo + (spec.stop[axis] - spec.start[axis])) for axis in range(3))
        output_values = {
            "shock_zone_mask": local_pass.shock_zone_mask[core_crop],
            "full_shock_mask": local_pass.full_shock_mask[core_crop],
            "shock_mask": local_pass.full_shock_mask[core_crop].copy(),
            "mach_temperature": local_pass.mach_fields.mach_temperature[core_crop],
            "mach_pressure": local_pass.mach_fields.mach_pressure[core_crop],
            "compression": local_pass.derived.compression[core_crop],
            "div_v": local_pass.derived.div_v[core_crop],
            "temperature": local_pass.derived.temperature[core_crop],
            "entropy": local_pass.derived.entropy[core_crop],
            "temperature_jump": local_pass.jumps.temperature_jump[core_crop],
            "pressure_jump": local_pass.jumps.pressure_jump[core_crop],
            "density_jump": local_pass.jumps.density_jump[core_crop],
            "normal_x": local_pass.normals.normal_x[core_crop],
            "normal_y": local_pass.normals.normal_y[core_crop],
            "normal_z": local_pass.normals.normal_z[core_crop],
        }
        for field_name in OUTPUT_FIELDS:
            write_chunk_file(output_path / field_name, spec, output_values[field_name])

        chunk_shock_mask = output_values["shock_mask"]
        shock_zone_cells += int(np.count_nonzero(output_values["shock_zone_mask"]))
        full_shock_cells += int(np.count_nonzero(output_values["full_shock_mask"]))
        mach_pressure_values.append(output_values["mach_pressure"][chunk_shock_mask])
        mach_temperature_values.append(output_values["mach_temperature"][chunk_shock_mask])

    report("[2/2] Writing chunked summary")
    summary = _make_chunked_summary(
        shape=shape,
        shock_zone_cells=shock_zone_cells,
        full_shock_cells=full_shock_cells,
        mach_pressure_values=mach_pressure_values,
        mach_temperature_values=mach_temperature_values,
        config=config,
        gamma=gamma,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if config.reduce_to_centers:
        summary = reduce_chunked_shock_outputs(output_path, config, progress=progress_callback is not None)
    return summary


def _build_field_readers(input_root: Path) -> dict[str, _FieldReader]:
    readers: dict[str, _FieldReader] = {}
    base_starts_stops: list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None
    for logical_name, field_dir_name in INPUT_FIELD_DIRS.items():
        field_dir = input_root / field_dir_name
        specs = load_chunk_specs(field_dir)
        with np.load(specs[0].path) as first_chunk:
            dtype = first_chunk["values"].dtype
        starts_stops = [(spec.start, spec.stop) for spec in specs]
        if base_starts_stops is None:
            base_starts_stops = starts_stops
        elif starts_stops != base_starts_stops:
            raise ValueError(f"Chunk layout mismatch in {field_dir}.")
        readers[logical_name] = _FieldReader(field_dir=field_dir, specs=specs, dtype=dtype)
    return readers


def _read_local_cube(
    readers: dict[str, _FieldReader],
    spec: ChunkSpec,
    halo: int,
    *,
    dx: float,
    dy: float,
    dz: float,
    gamma: float,
) -> FluidCube:
    request_slices = tuple(
        slice(spec.start[axis] - halo, spec.stop[axis] + halo)
        for axis in range(3)
    )
    return FluidCube(
        rho=readers["rho"].read_region(request_slices),
        pressure=readers["pressure"].read_region(request_slices),
        vx=readers["vx"].read_region(request_slices),
        vy=readers["vy"].read_region(request_slices),
        vz=readers["vz"].read_region(request_slices),
        dx=dx,
        dy=dy,
        dz=dz,
        gamma=gamma,
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
    specs: list[ChunkSpec],
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
        with np.load(spec.path) as chunk:
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


def _make_chunked_summary(
    *,
    shape: tuple[int, int, int],
    shock_zone_cells: int,
    full_shock_cells: int,
    mach_pressure_values: list[np.ndarray],
    mach_temperature_values: list[np.ndarray],
    config: ShockFinderConfig,
    gamma: float,
) -> dict[str, object]:
    total_cells = int(np.prod(shape))
    pressure_finite = _concatenate_finite(mach_pressure_values)
    temperature_finite = _concatenate_finite(mach_temperature_values)
    return {
        "grid_shape": list(shape),
        "total_cells": total_cells,
        "shock_zone_cells": shock_zone_cells,
        "shock_cells": full_shock_cells,
        "shock_fraction": full_shock_cells / total_cells,
        "full_shock_cells": full_shock_cells,
        "full_shock_fraction": full_shock_cells / total_cells,
        "reduce_to_centers": False,
        "requested_reduce_to_centers": config.reduce_to_centers,
        "center_score": config.center_score,
        "sampling_method": config.sampling_method,
        "shock_width_cells": config.shock_width_cells,
        "min_mach": config.min_mach,
        "gamma": gamma,
        "n_connected_components": None,
        "mask_semantics": "shock_mask equals full_shock_mask in chunked streamed mode",
        "mach_pressure_min": _finite_stat(pressure_finite, np.min),
        "mach_pressure_median": _finite_stat(pressure_finite, np.median),
        "mach_pressure_max": _finite_stat(pressure_finite, np.max),
        "mach_temperature_min": _finite_stat(temperature_finite, np.min),
        "mach_temperature_median": _finite_stat(temperature_finite, np.median),
        "mach_temperature_max": _finite_stat(temperature_finite, np.max),
        "full_mach_pressure_min": _finite_stat(pressure_finite, np.min),
        "full_mach_pressure_median": _finite_stat(pressure_finite, np.median),
        "full_mach_pressure_max": _finite_stat(pressure_finite, np.max),
        "full_mach_temperature_min": _finite_stat(temperature_finite, np.min),
        "full_mach_temperature_median": _finite_stat(temperature_finite, np.median),
        "full_mach_temperature_max": _finite_stat(temperature_finite, np.max),
    }


def _concatenate_finite(values: list[np.ndarray]) -> np.ndarray:
    finite_chunks = [chunk[np.isfinite(chunk)] for chunk in values if chunk.size > 0]
    if not finite_chunks:
        return np.array([], dtype=float)
    return np.concatenate(finite_chunks)


def _finite_stat(values: np.ndarray, reducer) -> float:
    if values.size == 0:
        return float("nan")
    return float(reducer(values))

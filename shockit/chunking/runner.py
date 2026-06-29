"""Explicit chunked runner for streamed shock finding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from ..config import ShockFinderConfig
from ..fields import FluidCube
from ..finder import analyze_cube_pass
from ..result import RESULT_FIELD_NAMES
from .centers import finalize_chunked_shock_outputs
from .reader import ChunkedInputStore
from .writer import ChunkedOutputStore

ProgressCallback = Callable[[str], None]
ProgressReporter = ProgressCallback | Literal[True, False] | None

OUTPUT_FIELDS = RESULT_FIELD_NAMES


def required_halo(config: ShockFinderConfig) -> int:
    """Return the halo width required by the current algorithm."""

    gradient_halo = 1
    sampling_halo = config.shock_width_cells
    return max(gradient_halo, sampling_halo)


def run_chunked_shock_finder(
    input_store: ChunkedInputStore,
    output_store: ChunkedOutputStore,
    *,
    config: ShockFinderConfig,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    gamma: float = 5.0 / 3.0,
    progress: ProgressReporter = True,
) -> dict[str, object]:
    """Run the shock finder explicitly on chunked input/output stores."""

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

    layout = input_store.layout
    halo = required_halo(config)
    total_steps = 2 if config.reduce_to_centers else 1
    report(f"[1/{total_steps}] Processing {len(layout.specs)} chunk files with halo width {halo}")

    primitive_readers = {
        name: input_store.field_reader(name)
        for name in ("rho", "pressure", "vx", "vy", "vz")
    }
    shock_zone_cells = 0
    full_shock_cells = 0
    full_mach_pressure_values: list[np.ndarray] = []
    full_mach_temperature_values: list[np.ndarray] = []

    for chunk_index, spec in enumerate(layout.specs, start=1):
        report(f"[1/{total_steps}] Chunk {chunk_index}/{len(layout.specs)} {spec.start}->{spec.stop}")
        local_cube = FluidCube(
            rho=primitive_readers["rho"].read_with_halo(spec, halo),
            pressure=primitive_readers["pressure"].read_with_halo(spec, halo),
            vx=primitive_readers["vx"].read_with_halo(spec, halo),
            vy=primitive_readers["vy"].read_with_halo(spec, halo),
            vz=primitive_readers["vz"].read_with_halo(spec, halo),
            dx=dx,
            dy=dy,
            dz=dz,
            gamma=gamma,
        )
        local_pass = analyze_cube_pass(local_cube, config)
        core_crop = tuple(slice(halo, halo + size) for size in spec.core_shape)
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
            output_store.write_core(field_name, spec, output_values[field_name])

        full_mask = output_values["full_shock_mask"].astype(bool)
        shock_zone_cells += int(np.count_nonzero(output_values["shock_zone_mask"]))
        full_shock_cells += int(np.count_nonzero(full_mask))
        full_mach_pressure_values.append(output_values["mach_pressure"][full_mask])
        full_mach_temperature_values.append(output_values["mach_temperature"][full_mask])

    base_summary = _base_summary(
        shape=layout.shape,
        shock_zone_cells=shock_zone_cells,
        full_shock_cells=full_shock_cells,
        full_mach_pressure_values=full_mach_pressure_values,
        full_mach_temperature_values=full_mach_temperature_values,
        config=config,
        gamma=gamma,
    )
    if not config.reduce_to_centers:
        output_store.write_summary(base_summary)
        return base_summary
    return finalize_chunked_shock_outputs(
        output_store,
        config,
        base_summary,
        progress=progress,
    )


def _base_summary(
    *,
    shape: tuple[int, int, int],
    shock_zone_cells: int,
    full_shock_cells: int,
    full_mach_pressure_values: list[np.ndarray],
    full_mach_temperature_values: list[np.ndarray],
    config: ShockFinderConfig,
    gamma: float,
) -> dict[str, object]:
    total_cells = int(np.prod(shape))
    mach_pressure = _concat(full_mach_pressure_values)
    mach_temperature = _concat(full_mach_temperature_values)
    return {
        "grid_shape": shape,
        "total_cells": total_cells,
        "shock_zone_cells": shock_zone_cells,
        "shock_cells": full_shock_cells,
        "shock_fraction": full_shock_cells / total_cells,
        "full_shock_cells": full_shock_cells,
        "full_shock_fraction": full_shock_cells / total_cells,
        "n_connected_components": 0,
        "mask_semantics": "shock_mask matches full_shock_mask; center reduction disabled",
        "mach_pressure_min": _finite_stat(mach_pressure, np.min),
        "mach_pressure_median": _finite_stat(mach_pressure, np.median),
        "mach_pressure_max": _finite_stat(mach_pressure, np.max),
        "mach_temperature_min": _finite_stat(mach_temperature, np.min),
        "mach_temperature_median": _finite_stat(mach_temperature, np.median),
        "mach_temperature_max": _finite_stat(mach_temperature, np.max),
        "full_mach_pressure_min": _finite_stat(mach_pressure, np.min),
        "full_mach_pressure_median": _finite_stat(mach_pressure, np.median),
        "full_mach_pressure_max": _finite_stat(mach_pressure, np.max),
        "full_mach_temperature_min": _finite_stat(mach_temperature, np.min),
        "full_mach_temperature_median": _finite_stat(mach_temperature, np.median),
        "full_mach_temperature_max": _finite_stat(mach_temperature, np.max),
        "min_mach": config.min_mach,
        "gamma": gamma,
        "shock_width_cells": config.shock_width_cells,
        "reduce_to_centers": config.reduce_to_centers,
        "center_score": config.center_score,
        "sampling_method": config.sampling_method,
    }


def _concat(values: list[np.ndarray]) -> np.ndarray:
    non_empty = [array for array in values if array.size > 0]
    if not non_empty:
        return np.array([], dtype=float)
    return np.concatenate(non_empty)


def _finite_stat(values: np.ndarray, reducer) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))

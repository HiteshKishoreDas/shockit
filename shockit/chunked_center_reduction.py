"""Second-pass center reduction for chunked shock-finder outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunked import join_chunked_field, load_chunk_specs, write_chunk_file
from .config import ShockFinderConfig
from .derived import MachFields
from .masks import reduce_to_centers


def reduce_chunked_shock_outputs(
    output_root: str | Path,
    config: ShockFinderConfig,
    *,
    progress: bool = True,
) -> dict[str, object]:
    """Reduce chunked unreduced shock outputs to one center per global component."""

    output_path = Path(output_root)
    if progress:
        print("[1/2] Loading chunked full-shock fields for center reduction")

    full_shock_mask = join_chunked_field(output_path / "full_shock_mask").astype(bool)
    div_v = join_chunked_field(output_path / "div_v")
    mach_fields = MachFields(
        mach_pressure=join_chunked_field(output_path / "mach_pressure"),
        mach_temperature=join_chunked_field(output_path / "mach_temperature"),
    )

    if progress:
        print("[2/2] Reducing connected components and rewriting chunked shock_mask")

    shock_mask, n_connected_components = reduce_to_centers(
        full_shock_mask,
        div_v,
        mach_fields,
        center_score_mode=config.center_score,
    )
    for spec in load_chunk_specs(output_path / "full_shock_mask"):
        chunk_slice = tuple(slice(spec.start[axis], spec.stop[axis]) for axis in range(3))
        write_chunk_file(output_path / "shock_mask", spec, shock_mask[chunk_slice])

    summary_path = output_path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary.update(
        {
            "shock_cells": int(np.count_nonzero(shock_mask)),
            "shock_fraction": float(np.count_nonzero(shock_mask) / np.prod(shock_mask.shape)),
            "n_connected_components": int(n_connected_components),
            "reduce_to_centers": True,
            "mask_semantics": "shock_mask is center-reduced; full_shock_mask is unreduced",
            "mach_pressure_min": _finite_stat(mach_fields.mach_pressure[shock_mask], np.min),
            "mach_pressure_median": _finite_stat(mach_fields.mach_pressure[shock_mask], np.median),
            "mach_pressure_max": _finite_stat(mach_fields.mach_pressure[shock_mask], np.max),
            "mach_temperature_min": _finite_stat(mach_fields.mach_temperature[shock_mask], np.min),
            "mach_temperature_median": _finite_stat(mach_fields.mach_temperature[shock_mask], np.median),
            "mach_temperature_max": _finite_stat(mach_fields.mach_temperature[shock_mask], np.max),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _finite_stat(values: np.ndarray, reducer) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))

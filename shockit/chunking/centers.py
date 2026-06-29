"""Chunk-aware connected-component reduction for chunked outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import shutil
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import ShockFinderConfig
from ..derived import MachFields
from ..masks import center_score, reduce_to_centers
from .layout import ChunkLayout, ChunkSpec
from .reader import ChunkedInputStore
from .writer import ChunkedOutputStore, NpzChunkedOutput

ProgressCallback = Callable[[str], None]
ProgressReporter = ProgressCallback | bool | None


@dataclass(frozen=True)
class _LabelRef:
    grid_index: tuple[int, int, int]
    local_label: int


@dataclass(frozen=True)
class _LocalLabelRecord:
    ref: _LabelRef
    spec: ChunkSpec
    best_local_index: tuple[int, int, int]
    best_global_flat_index: int
    best_score: float
    best_mach_pressure: float
    best_mach_temperature: float


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[_LabelRef, _LabelRef] = {}

    def add(self, item: _LabelRef) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: _LabelRef) -> _LabelRef:
        parent = self.parent[item]
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, left: _LabelRef, right: _LabelRef) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def finalize_chunked_shock_outputs(
    output_store: ChunkedOutputStore,
    config: ShockFinderConfig,
    base_summary: dict[str, Any],
    *,
    progress: ProgressReporter = True,
) -> dict[str, Any]:
    """Finalize connected-component accounting and optional center reduction."""

    if progress is True or progress is None:
        progress_callback: ProgressCallback | None = print
    elif progress is False:
        progress_callback = None
    else:
        progress_callback = progress

    if progress_callback is not None:
        progress_callback("[2/2] Reducing connected components across chunk boundaries")

    fields = output_store
    labels_root = output_store.root / ".local_labels"
    shutil.rmtree(labels_root, ignore_errors=True)
    label_store = NpzChunkedOutput(labels_root, output_store.layout)
    label_records, union_find = _label_local_components(fields, label_store, config.center_score)
    n_connected_components = _merge_boundary_components(output_store.layout, union_find, label_store)

    summary = dict(base_summary)
    summary["n_connected_components"] = n_connected_components

    if not config.reduce_to_centers:
        summary.update(
            {
                "shock_cells": summary["full_shock_cells"],
                "shock_fraction": summary["full_shock_fraction"],
                "mask_semantics": "shock_mask matches full_shock_mask; center reduction disabled",
                "mach_pressure_min": summary["full_mach_pressure_min"],
                "mach_pressure_median": summary["full_mach_pressure_median"],
                "mach_pressure_max": summary["full_mach_pressure_max"],
                "mach_temperature_min": summary["full_mach_temperature_min"],
                "mach_temperature_median": summary["full_mach_temperature_median"],
                "mach_temperature_max": summary["full_mach_temperature_max"],
            }
        )
        output_store.write_summary(summary)
        shutil.rmtree(labels_root, ignore_errors=True)
        return summary

    centers = _select_global_centers(
        layout=output_store.layout,
        fields=fields,
        label_store=label_store,
        label_records=label_records,
        union_find=union_find,
        center_score_mode=config.center_score,
    )
    _write_center_mask(output_store, centers)
    mach_pressure_values = np.array([center[1] for center in centers], dtype=float)
    mach_temperature_values = np.array([center[2] for center in centers], dtype=float)
    summary.update(
        {
            "shock_cells": len(centers),
            "shock_fraction": len(centers) / summary["total_cells"],
            "mask_semantics": "shock_mask is center-reduced; full_shock_mask is unreduced",
            "mach_pressure_min": _finite_stat(mach_pressure_values, np.min),
            "mach_pressure_median": _finite_stat(mach_pressure_values, np.median),
            "mach_pressure_max": _finite_stat(mach_pressure_values, np.max),
            "mach_temperature_min": _finite_stat(mach_temperature_values, np.min),
            "mach_temperature_median": _finite_stat(mach_temperature_values, np.median),
            "mach_temperature_max": _finite_stat(mach_temperature_values, np.max),
        }
    )
    output_store.write_summary(summary)
    shutil.rmtree(labels_root, ignore_errors=True)
    return summary


def _label_local_components(
    fields: ChunkedInputStore,
    label_store: ChunkedOutputStore,
    center_score_mode: str,
) -> tuple[dict[_LabelRef, _LocalLabelRecord], _UnionFind]:
    records: dict[_LabelRef, _LocalLabelRecord] = {}
    union_find = _UnionFind()
    mask_reader = fields.field_reader("full_shock_mask")
    div_v_reader = fields.field_reader("div_v")
    pressure_reader = fields.field_reader("mach_pressure")
    temperature_reader = fields.field_reader("mach_temperature")

    for spec in fields.layout.specs:
        mask = mask_reader.read_core(spec).astype(bool)
        labels, count = ndimage.label(mask)
        label_store.write_core("labels", spec, labels.astype(np.int32))
        if count == 0:
            continue
        div_v = div_v_reader.read_core(spec)
        mach_fields = MachFields(
            mach_pressure=pressure_reader.read_core(spec),
            mach_temperature=temperature_reader.read_core(spec),
        )
        simple_score = _simple_score_field(div_v, mach_fields, center_score_mode)
        for local_label in range(1, count + 1):
            if simple_score is None:
                component = labels == local_label
                score = center_score(component, div_v, mach_fields, mode=center_score_mode)
                best_flat_index = int(np.argmax(score))
                best_local_index = np.unravel_index(best_flat_index, score.shape)
                best_score = float(score[best_local_index])
            else:
                best_local_index = tuple(
                    int(value)
                    for value in ndimage.maximum_position(simple_score, labels=labels, index=[local_label])[0]
                )
                best_score = float(simple_score[best_local_index])
            best_global_index = tuple(spec.start[axis] + best_local_index[axis] for axis in range(3))
            ref = _LabelRef(spec.grid_index, local_label)
            union_find.add(ref)
            records[ref] = _LocalLabelRecord(
                ref=ref,
                spec=spec,
                best_local_index=best_local_index,
                best_global_flat_index=int(np.ravel_multi_index(best_global_index, spec.full_shape)),
                best_score=best_score,
                best_mach_pressure=float(mach_fields.mach_pressure[best_local_index]),
                best_mach_temperature=float(mach_fields.mach_temperature[best_local_index]),
            )
    return records, union_find


def _merge_boundary_components(
    layout: ChunkLayout,
    union_find: _UnionFind,
    label_store: ChunkedOutputStore,
) -> int:
    label_reader = label_store.field_reader("labels")
    for axis, spec_a, spec_b in layout.iter_face_neighbor_pairs(periodic=True):
        labels_a = label_reader.read_core(spec_a)
        labels_b = label_reader.read_core(spec_b)
        face_a = np.take(labels_a, indices=-1, axis=axis)
        face_b = np.take(labels_b, indices=0, axis=axis)
        overlaps = (face_a > 0) & (face_b > 0)
        if not np.any(overlaps):
            continue
        pairs = np.unique(np.stack([face_a[overlaps], face_b[overlaps]], axis=1), axis=0)
        for label_a, label_b in pairs:
            union_find.union(
                _LabelRef(spec_a.grid_index, int(label_a)),
                _LabelRef(spec_b.grid_index, int(label_b)),
            )
    return len({union_find.find(item) for item in union_find.parent})


def _select_global_centers(
    layout: ChunkLayout,
    fields: ChunkedInputStore,
    label_store: ChunkedOutputStore,
    label_records: dict[_LabelRef, _LocalLabelRecord],
    union_find: _UnionFind,
    center_score_mode: str,
) -> list[tuple[tuple[int, int, int], float, float]]:
    groups: dict[_LabelRef, list[_LabelRef]] = {}
    for ref in label_records:
        groups.setdefault(union_find.find(ref), []).append(ref)

    if center_score_mode != "combined":
        label_reader = label_store.field_reader("labels")
        div_v_reader = fields.field_reader("div_v")
        pressure_reader = fields.field_reader("mach_pressure")
        temperature_reader = fields.field_reader("mach_temperature")
        centers: list[tuple[tuple[int, int, int], float, float]] = []
        for refs in groups.values():
            origin, group_mask, group_div_v, group_mach_fields = _assemble_group_fields(
                refs,
                label_records,
                label_reader,
                div_v_reader,
                pressure_reader,
                temperature_reader,
            )
            _, local_component_count = ndimage.label(group_mask)
            if local_component_count == 1:
                reduced, _ = reduce_to_centers(
                    group_mask,
                    group_div_v,
                    group_mach_fields,
                    center_score_mode=center_score_mode,
                )
                best_local_index = tuple(int(value) for value in np.argwhere(reduced)[0])
                best_global_index = tuple(origin[axis] + best_local_index[axis] for axis in range(3))
                centers.append(
                    (
                        best_global_index,
                        float(group_mach_fields.mach_pressure[best_local_index]),
                        float(group_mach_fields.mach_temperature[best_local_index]),
                    )
                )
                continue
            best_candidate: tuple[float, int, tuple[int, int, int], float, float] | None = None
            for ref in refs:
                record = label_records[ref]
                labels = label_reader.read_core(record.spec)
                component = labels == ref.local_label
                div_v = div_v_reader.read_core(record.spec)
                mach_fields = MachFields(
                    mach_pressure=pressure_reader.read_core(record.spec),
                    mach_temperature=temperature_reader.read_core(record.spec),
                )
                score = center_score(component, div_v, mach_fields, mode=center_score_mode)
                for local_index in zip(*np.where(component), strict=False):
                    local_score = float(score[local_index])
                    global_index = tuple(record.spec.start[axis] + local_index[axis] for axis in range(3))
                    global_flat = int(np.ravel_multi_index(global_index, layout.shape))
                    tie_break_flat = -global_flat if local_score > 0.0 else global_flat
                    candidate = (
                        -local_score,
                        tie_break_flat,
                        global_index,
                        float(mach_fields.mach_pressure[local_index]),
                        float(mach_fields.mach_temperature[local_index]),
                    )
                    if best_candidate is None or candidate[:2] < best_candidate[:2]:
                        best_candidate = candidate
            assert best_candidate is not None
            centers.append((best_candidate[2], best_candidate[3], best_candidate[4]))
        return centers

    label_reader = label_store.field_reader("labels")
    div_v_reader = fields.field_reader("div_v")
    pressure_reader = fields.field_reader("mach_pressure")
    temperature_reader = fields.field_reader("mach_temperature")
    centers = []
    for refs in groups.values():
        compression_min, compression_max, mach_min, mach_max = _combined_score_ranges(
            refs,
            label_records,
            label_reader,
            div_v_reader,
            pressure_reader,
            temperature_reader,
        )
        best_candidate: tuple[float, int, tuple[int, int, int], float, float] | None = None
        for ref in refs:
            record = label_records[ref]
            labels = label_reader.read_core(record.spec)
            component = labels == ref.local_label
            div_v = div_v_reader.read_core(record.spec)
            mach_pressure = pressure_reader.read_core(record.spec)
            mach_temperature = temperature_reader.read_core(record.spec)
            mach = np.maximum(
                np.nan_to_num(mach_pressure, nan=-np.inf),
                np.nan_to_num(mach_temperature, nan=-np.inf),
            )
            score = _combined_component_score(component, -div_v, mach, compression_min, compression_max, mach_min, mach_max)
            for local_index in zip(*np.where(component), strict=False):
                local_score = float(score[local_index])
                global_index = tuple(record.spec.start[axis] + local_index[axis] for axis in range(3))
                global_flat = int(np.ravel_multi_index(global_index, layout.shape))
                candidate = (
                    -local_score,
                    global_flat,
                    global_index,
                    float(mach_pressure[local_index]),
                    float(mach_temperature[local_index]),
                )
                if best_candidate is None or candidate[:2] < best_candidate[:2]:
                    best_candidate = candidate
        assert best_candidate is not None
        centers.append((best_candidate[2], best_candidate[3], best_candidate[4]))
    return centers


def build_base_summary_from_output_store(
    output_store: ChunkedOutputStore,
    config: ShockFinderConfig,
    *,
    gamma: float,
) -> dict[str, Any]:
    """Rebuild base chunked summary stats from stored output fields."""

    shock_zone_reader = output_store.field_reader("shock_zone_mask")
    full_shock_reader = output_store.field_reader("full_shock_mask")
    pressure_reader = output_store.field_reader("mach_pressure")
    temperature_reader = output_store.field_reader("mach_temperature")

    shock_zone_cells = 0
    full_shock_cells = 0
    mach_pressure_values: list[np.ndarray] = []
    mach_temperature_values: list[np.ndarray] = []
    for spec in output_store.layout.specs:
        shock_zone = shock_zone_reader.read_core(spec).astype(bool)
        full_shock = full_shock_reader.read_core(spec).astype(bool)
        mach_pressure = pressure_reader.read_core(spec)
        mach_temperature = temperature_reader.read_core(spec)
        shock_zone_cells += int(np.count_nonzero(shock_zone))
        full_shock_cells += int(np.count_nonzero(full_shock))
        mach_pressure_values.append(mach_pressure[full_shock])
        mach_temperature_values.append(mach_temperature[full_shock])

    total_cells = int(np.prod(output_store.layout.shape))
    mach_pressure = _concat(mach_pressure_values)
    mach_temperature = _concat(mach_temperature_values)
    return {
        "grid_shape": output_store.layout.shape,
        "total_cells": total_cells,
        "shock_zone_cells": shock_zone_cells,
        "shock_cells": full_shock_cells,
        "shock_fraction": full_shock_cells / total_cells,
        "full_shock_cells": full_shock_cells,
        "full_shock_fraction": full_shock_cells / total_cells,
        "n_connected_components": 0,
        "mask_semantics": "shock_mask matches full_shock_mask until center reduction runs",
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


def _combined_score_ranges(
    refs: list[_LabelRef],
    label_records: dict[_LabelRef, _LocalLabelRecord],
    label_reader,
    div_v_reader,
    pressure_reader,
    temperature_reader,
) -> tuple[float, float, float, float]:
    compression_values: list[np.ndarray] = []
    mach_values: list[np.ndarray] = []
    for ref in refs:
        spec = label_records[ref].spec
        labels = label_reader.read_core(spec)
        component = labels == ref.local_label
        compression_values.append((-div_v_reader.read_core(spec))[component])
        mach_pressure = pressure_reader.read_core(spec)
        mach_temperature = temperature_reader.read_core(spec)
        combined_mach = np.maximum(
            np.nan_to_num(mach_pressure, nan=-np.inf),
            np.nan_to_num(mach_temperature, nan=-np.inf),
        )
        finite_mach = combined_mach[component]
        mach_values.append(finite_mach[np.isfinite(finite_mach)])

    compression_concat = np.concatenate(compression_values) if compression_values else np.array([], dtype=float)
    finite_compression = compression_concat[np.isfinite(compression_concat)]
    mach_concat = np.concatenate(mach_values) if mach_values else np.array([], dtype=float)
    compression_min = float(np.min(finite_compression)) if finite_compression.size else 0.0
    compression_max = float(np.max(finite_compression)) if finite_compression.size else 0.0
    mach_min = float(np.min(mach_concat)) if mach_concat.size else 0.0
    mach_max = float(np.max(mach_concat)) if mach_concat.size else 0.0
    return compression_min, compression_max, mach_min, mach_max


def _assemble_group_fields(
    refs: list[_LabelRef],
    label_records: dict[_LabelRef, _LocalLabelRecord],
    label_reader,
    div_v_reader,
    pressure_reader,
    temperature_reader,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, MachFields]:
    start = tuple(min(label_records[ref].spec.start[axis] for ref in refs) for axis in range(3))
    stop = tuple(max(label_records[ref].spec.stop[axis] for ref in refs) for axis in range(3))
    shape = tuple(stop[axis] - start[axis] for axis in range(3))
    mask = np.zeros(shape, dtype=bool)
    div_v = np.zeros(shape, dtype=float)
    mach_pressure = np.full(shape, np.nan, dtype=float)
    mach_temperature = np.full(shape, np.nan, dtype=float)

    for ref in refs:
        spec = label_records[ref].spec
        local_slices = tuple(
            slice(spec.start[axis] - start[axis], spec.stop[axis] - start[axis])
            for axis in range(3)
        )
        labels = label_reader.read_core(spec)
        component = labels == ref.local_label
        div_v_chunk = div_v_reader.read_core(spec)
        mach_pressure_chunk = pressure_reader.read_core(spec)
        mach_temperature_chunk = temperature_reader.read_core(spec)
        mask_region = mask[local_slices]
        div_v_region = div_v[local_slices]
        mach_pressure_region = mach_pressure[local_slices]
        mach_temperature_region = mach_temperature[local_slices]
        mask_region[component] = True
        div_v_region[component] = div_v_chunk[component]
        mach_pressure_region[component] = mach_pressure_chunk[component]
        mach_temperature_region[component] = mach_temperature_chunk[component]

    return start, mask, div_v, MachFields(mach_pressure=mach_pressure, mach_temperature=mach_temperature)


def _combined_component_score(
    component: np.ndarray,
    compression: np.ndarray,
    mach: np.ndarray,
    compression_min: float,
    compression_max: float,
    mach_min: float,
    mach_max: float,
) -> np.ndarray:
    score = np.full(component.shape, -np.inf, dtype=float)
    if not np.any(component):
        return score
    compression_norm = _normalize_values(compression[component], compression_min, compression_max)
    if mach_max <= mach_min:
        mach_norm = np.zeros(np.count_nonzero(component), dtype=float)
    else:
        mach_component = mach[component]
        mach_norm = np.zeros_like(mach_component, dtype=float)
        finite = np.isfinite(mach_component)
        mach_norm[finite] = (mach_component[finite] - mach_min) / (mach_max - mach_min)
    score[component] = compression_norm + mach_norm
    return score


def _normalize_values(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.ones(values.shape, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _write_center_mask(
    output_store: ChunkedOutputStore,
    centers: list[tuple[tuple[int, int, int], float, float]],
) -> None:
    centers_by_grid: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for center_index, _, _ in centers:
        for spec in output_store.layout.specs:
            if all(spec.start[axis] <= center_index[axis] < spec.stop[axis] for axis in range(3)):
                centers_by_grid.setdefault(spec.grid_index, []).append(
                    tuple(center_index[axis] - spec.start[axis] for axis in range(3))
                )
                break

    for spec in output_store.layout.specs:
        mask = np.zeros(spec.core_shape, dtype=bool)
        for local_index in centers_by_grid.get(spec.grid_index, []):
            mask[local_index] = True
        output_store.write_core("shock_mask", spec, mask)


def _finite_stat(values: np.ndarray, reducer) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def _concat(values: list[np.ndarray]) -> np.ndarray:
    non_empty = [array for array in values if array.size > 0]
    if not non_empty:
        return np.array([], dtype=float)
    return np.concatenate(non_empty)


def _simple_score_field(
    div_v: np.ndarray,
    mach_fields: MachFields,
    mode: str,
) -> np.ndarray | None:
    if mode == "compression":
        return -div_v
    if mode == "mach_pressure":
        return np.nan_to_num(mach_fields.mach_pressure, nan=-np.inf)
    if mode == "mach_temperature":
        return np.nan_to_num(mach_fields.mach_temperature, nan=-np.inf)
    return None

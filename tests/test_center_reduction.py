from __future__ import annotations

import numpy as np

from shockit.derived import MachFields
from shockit.masks import center_score, reduce_to_centers


def test_connected_region_reduces_to_one_cell() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    div_v = np.zeros(mask.shape)
    div_v[2, 2, 2] = -5.0
    mach_fields = MachFields(
        mach_pressure=np.ones(mask.shape),
        mach_temperature=np.ones(mask.shape),
    )
    reduced, count = reduce_to_centers(mask, div_v, mach_fields, center_score_mode="compression")
    assert count == 1
    assert np.count_nonzero(reduced) == 1
    assert reduced[2, 2, 2]


def test_center_score_mach_pressure_selects_highest_mach() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 1, 2] = True
    div_v = np.zeros(component.shape)
    mach_fields = MachFields(
        mach_pressure=np.zeros(component.shape),
        mach_temperature=np.full(component.shape, np.nan),
    )
    mach_fields.mach_pressure[1, 1, 1] = 2.0
    mach_fields.mach_pressure[1, 1, 2] = 3.0
    score = center_score(component, div_v, mach_fields, mode="mach_pressure")
    assert np.argmax(score) == np.ravel_multi_index((1, 1, 2), component.shape)


def test_center_score_compression_selects_strongest_compression() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 2, 1] = True
    div_v = np.zeros(component.shape)
    div_v[1, 1, 1] = -1.0
    div_v[1, 2, 1] = -4.0
    mach_fields = MachFields(
        mach_pressure=np.full(component.shape, np.nan),
        mach_temperature=np.full(component.shape, np.nan),
    )
    score = center_score(component, div_v, mach_fields, mode="compression")
    assert np.argmax(score) == np.ravel_multi_index((1, 2, 1), component.shape)


def test_combined_score_handles_nan_mach_and_stays_inside_component() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[1, 1, 2] = True
    div_v = np.zeros(component.shape)
    div_v[1, 1, 1] = -2.0
    div_v[1, 1, 2] = -1.0
    mach_fields = MachFields(
        mach_pressure=np.full(component.shape, np.nan),
        mach_temperature=np.full(component.shape, np.nan),
    )
    score = center_score(component, div_v, mach_fields, mode="combined")
    best = np.unravel_index(int(np.argmax(score)), score.shape)
    assert component[best]
    assert np.isfinite(score[best])
    assert np.all(score[~component] == -np.inf)

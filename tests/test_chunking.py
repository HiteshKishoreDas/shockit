from __future__ import annotations

from shockit import ShockFinderConfig
from shockit.chunking.layout import iter_balanced_slices
from shockit.chunking.runner import required_halo


def test_required_halo_nearest_axis_uses_sampling_width() -> None:
    assert required_halo(ShockFinderConfig(sampling_method="nearest_axis", shock_width_cells=1)) == 1
    assert required_halo(ShockFinderConfig(sampling_method="nearest_axis", shock_width_cells=3)) == 3


def test_required_halo_trilinear_follows_same_rule_for_now() -> None:
    assert required_halo(ShockFinderConfig(sampling_method="trilinear", shock_width_cells=1)) == 1
    assert required_halo(ShockFinderConfig(sampling_method="trilinear", shock_width_cells=3)) == 3


def test_chunk_slices_are_balanced_near_target_size() -> None:
    slices = list(iter_balanced_slices((300, 260, 129), 128))
    axis_lengths = [{slc[axis].stop - slc[axis].start for slc in slices} for axis in range(3)]

    assert axis_lengths[0] == {100}
    assert axis_lengths[1] == {86, 87}
    assert axis_lengths[2] == {64, 65}


def test_chunk_slices_support_per_axis_chunk_shapes() -> None:
    slices = list(iter_balanced_slices((18, 18, 18), (8, 6, 5)))
    axis_lengths = [{slc[axis].stop - slc[axis].start for slc in slices} for axis in range(3)]

    assert axis_lengths[0] == {6}
    assert axis_lengths[1] == {6}
    assert axis_lengths[2] == {4, 5}

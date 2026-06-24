from __future__ import annotations

import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from tests.helpers import make_planar_shock_cube


def test_masks_are_nested_when_center_reduction_enabled() -> None:
    cube = make_planar_shock_cube()
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=True)).find(cube)
    assert np.all(result.full_shock_mask <= result.shock_zone_mask)
    assert np.all(result.shock_mask <= result.full_shock_mask)


def test_shock_mask_equals_full_mask_without_reduction() -> None:
    cube = make_planar_shock_cube()
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(cube)
    np.testing.assert_array_equal(result.shock_mask, result.full_shock_mask)

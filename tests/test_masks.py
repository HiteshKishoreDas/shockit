from __future__ import annotations

import numpy as np

from shockit import ShockFinder, ShockFinderConfig
from shockit.derived import MachFields
from shockit.masks import build_final_shock_mask
from shockit.sampling import SampledJumps
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


def test_full_shock_mask_rejects_density_jumps_above_physical_limit() -> None:
    shock_zone_mask = np.ones((1, 1, 1), dtype=bool)
    jumps = SampledJumps(
        pressure_jump=np.array([[[4.0]]]),
        temperature_jump=np.array([[[1.0]]]),
        density_jump=np.array([[[5.0]]]),
        pressure_up=np.array([[[1.0]]]),
        pressure_down=np.array([[[4.0]]]),
        temperature_up=np.array([[[1.0]]]),
        temperature_down=np.array([[[1.0]]]),
        density_up=np.array([[[5.0]]]),
        density_down=np.array([[[25.0]]]),
    )
    mach_fields = MachFields(
        mach_pressure=np.array([[[2.0]]]),
        mach_temperature=np.array([[[2.0]]]),
    )

    full_shock_mask = build_final_shock_mask(
        shock_zone_mask,
        jumps,
        mach_fields,
        ShockFinderConfig(),
        gamma=5.0 / 3.0,
    )

    assert not full_shock_mask[0, 0, 0]


def test_full_shock_mask_rejects_inconsistent_jump_triplets() -> None:
    shock_zone_mask = np.ones((1, 1, 1), dtype=bool)
    jumps = SampledJumps(
        pressure_jump=np.array([[[4.0]]]),
        temperature_jump=np.array([[[3.0]]]),
        density_jump=np.array([[[2.0]]]),
        pressure_up=np.array([[[1.0]]]),
        pressure_down=np.array([[[4.0]]]),
        temperature_up=np.array([[[1.0]]]),
        temperature_down=np.array([[[3.0]]]),
        density_up=np.array([[[2.0]]]),
        density_down=np.array([[[4.0]]]),
    )
    mach_fields = MachFields(
        mach_pressure=np.array([[[2.0]]]),
        mach_temperature=np.array([[[2.0]]]),
    )

    full_shock_mask = build_final_shock_mask(
        shock_zone_mask,
        jumps,
        mach_fields,
        ShockFinderConfig(),
        gamma=5.0 / 3.0,
    )

    assert not full_shock_mask[0, 0, 0]


def test_full_shock_mask_optionally_rejects_upstream_values_below_floors() -> None:
    shock_zone_mask = np.ones((1, 1, 1), dtype=bool)
    jumps = SampledJumps(
        pressure_jump=np.array([[[4.0]]]),
        temperature_jump=np.array([[[2.0]]]),
        density_jump=np.array([[[2.0]]]),
        pressure_up=np.array([[[0.05]]]),
        pressure_down=np.array([[[0.2]]]),
        temperature_up=np.array([[[0.1]]]),
        temperature_down=np.array([[[0.2]]]),
        density_up=np.array([[[0.1]]]),
        density_down=np.array([[[0.2]]]),
    )
    mach_fields = MachFields(
        mach_pressure=np.array([[[2.0]]]),
        mach_temperature=np.array([[[2.0]]]),
    )

    full_shock_mask = build_final_shock_mask(
        shock_zone_mask,
        jumps,
        mach_fields,
        ShockFinderConfig(
            upstream_pressure_floor=0.1,
            upstream_temperature_floor=0.2,
            upstream_density_floor=0.2,
        ),
        gamma=5.0 / 3.0,
    )

    assert not full_shock_mask[0, 0, 0]

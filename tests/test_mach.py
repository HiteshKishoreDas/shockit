import math

import pytest

from shockit.mach import (
    mach_from_pressure_jump,
    mach_from_temperature_jump,
    pressure_jump_from_mach,
    temperature_jump_from_mach,
)


@pytest.mark.parametrize("mach", [1.1, 1.5, 2.0, 5.0, 10.0])
def test_mach_inversion(mach: float, test_plotter) -> None:
    gamma = 5.0 / 3.0
    pressure_jump = pressure_jump_from_mach(mach, gamma)
    recovered_pressure_mach = mach_from_pressure_jump(pressure_jump, gamma)
    assert recovered_pressure_mach == pytest.approx(mach, rel=1e-10)

    temperature_jump = temperature_jump_from_mach(mach, gamma)
    recovered_temperature_mach = mach_from_temperature_jump(temperature_jump, gamma)
    assert recovered_temperature_mach == pytest.approx(mach, rel=1e-6)
    test_plotter.save_line(
        f"mach_temperature_jump_mach_{str(mach).replace('.', '_')}",
        [1.0, temperature_jump],
        [1.0, recovered_temperature_mach],
        title=f"Temperature-jump inversion at M={mach}",
        xlabel="Temperature jump",
        ylabel="Recovered Mach",
        label="Recovered Mach",
    )


def test_invalid_jumps_return_nan() -> None:
    gamma = 5.0 / 3.0
    assert math.isnan(mach_from_pressure_jump(1.0, gamma))
    assert math.isnan(mach_from_temperature_jump(1.0, gamma))

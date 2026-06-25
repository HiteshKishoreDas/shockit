from __future__ import annotations

import numpy as np

from shockit.derived import compute_mach_fields
from shockit.sampling import SampledJumps


def test_compute_mach_fields_skips_temperature_inversion_outside_candidate_mask(monkeypatch) -> None:
    shape = (2, 2, 2)
    temperature_jump = np.full(shape, 2.0)
    pressure_jump = np.full(shape, 2.0)
    candidate_mask = np.zeros(shape, dtype=bool)
    candidate_mask[0, 0, 0] = True
    candidate_mask[1, 1, 1] = True
    seen: list[float] = []

    def fake_mach_from_temperature_jump(value: float, gamma: float, mach_max: float) -> float:
        seen.append(value)
        return 3.0

    monkeypatch.setattr("shockit.derived.mach_from_temperature_jump", fake_mach_from_temperature_jump)

    fields = compute_mach_fields(
        SampledJumps(
            pressure_jump=pressure_jump,
            temperature_jump=temperature_jump,
            density_jump=np.full(shape, 2.0),
            pressure_up=np.ones(shape),
            pressure_down=np.full(shape, 2.0),
        ),
        gamma=5.0 / 3.0,
        mach_max=1000.0,
        candidate_mask=candidate_mask,
    )

    assert len(seen) == int(np.count_nonzero(candidate_mask))
    assert np.isfinite(fields.mach_temperature[candidate_mask]).all()
    assert np.isfinite(fields.mach_pressure[candidate_mask]).all()
    assert np.isnan(fields.mach_pressure[~candidate_mask]).all()
    assert np.isnan(fields.mach_temperature[~candidate_mask]).all()

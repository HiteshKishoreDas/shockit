"""Main shock finding API."""

from __future__ import annotations

from .config import ShockFinderConfig
from .derived import compute_derived_fields, compute_gradients, compute_mach_fields, compute_normals
from .fields import FluidCube
from .masks import build_final_shock_mask, build_shock_zone_mask, reduce_to_centers
from .result import ShockFinderResult
from .sampling import sample_jumps
from .summary import make_summary


class ShockFinder:
    """Skillman-style shock finder for uniform fluid cubes."""

    def __init__(self, config: ShockFinderConfig | None = None) -> None:
        self.config = config or ShockFinderConfig()

    def find(self, cube: FluidCube) -> ShockFinderResult:
        """Run the shock finder as a clear sequence of reviewable steps."""

        derived = compute_derived_fields(cube)
        gradients = compute_gradients(cube, derived.temperature, derived.entropy)
        shock_zone_mask = build_shock_zone_mask(derived, gradients, self.config)
        normals = compute_normals(gradients, self.config.normal_field)
        jumps = sample_jumps(
            pressure=cube.pressure,
            temperature=derived.temperature,
            rho=cube.rho,
            nx=normals.normal_x,
            ny=normals.normal_y,
            nz=normals.normal_z,
            width=self.config.shock_width_cells,
            method=self.config.sampling_method,
            candidate_mask=shock_zone_mask,
        )
        mach_fields = compute_mach_fields(
            jumps,
            cube.gamma,
            mach_max=self.config.mach_max,
            candidate_mask=shock_zone_mask,
        )
        full_shock_mask = build_final_shock_mask(shock_zone_mask, jumps, mach_fields, self.config)
        if self.config.reduce_to_centers:
            shock_mask, _ = reduce_to_centers(
                full_shock_mask,
                derived.div_v,
                mach_fields,
                center_score_mode=self.config.center_score,
            )
        else:
            shock_mask = full_shock_mask.copy()

        summary = make_summary(
            cube=cube,
            shock_zone_mask=shock_zone_mask,
            full_shock_mask=full_shock_mask,
            shock_mask=shock_mask,
            mach_fields=mach_fields,
            config=self.config,
        )
        return ShockFinderResult(
            shock_mask=shock_mask,
            shock_zone_mask=shock_zone_mask,
            mach_temperature=mach_fields.mach_temperature,
            mach_pressure=mach_fields.mach_pressure,
            compression=derived.compression,
            div_v=derived.div_v,
            temperature=derived.temperature,
            entropy=derived.entropy,
            temperature_jump=jumps.temperature_jump,
            pressure_jump=jumps.pressure_jump,
            density_jump=jumps.density_jump,
            normal_x=normals.normal_x,
            normal_y=normals.normal_y,
            normal_z=normals.normal_z,
            summary=summary,
            full_shock_mask=full_shock_mask,
        )

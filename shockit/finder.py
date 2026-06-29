"""Main shock finding API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .config import ShockFinderConfig
from .derived import (
    DerivedFields,
    MachFields,
    NormalFields,
    compute_derived_fields,
    compute_gradients,
    compute_mach_fields,
    compute_normals,
)
from .fields import FluidCube
from .masks import build_final_shock_mask, build_shock_zone_mask, reduce_to_centers
from .result import ChunkedShockFinderResult, RESULT_FIELD_NAMES, ShockFinderResult
from .sampling import SampledJumps, sample_jumps
from .storage import ChunkedFieldReference
from .summary import make_summary

ProgressCallback = Callable[[str], None]
ProgressReporter = ProgressCallback | Literal[True, False] | None


@dataclass
class _FinderPass:
    derived: DerivedFields
    normals: NormalFields
    shock_zone_mask: np.ndarray
    jumps: SampledJumps
    mach_fields: MachFields
    full_shock_mask: np.ndarray


class ShockFinder:
    """Skillman-style shock finder for uniform fluid cubes."""

    def __init__(self, config: ShockFinderConfig | None = None) -> None:
        self.config = config or ShockFinderConfig()

    def find(
        self,
        cube: FluidCube,
        progress: ProgressReporter = True,
        output: str | Path | None = None,
    ) -> ShockFinderResult | ChunkedShockFinderResult:
        """Run the shock finder on either in-memory or chunked primitive fields."""

        if not isinstance(cube, FluidCube):
            raise TypeError("ShockFinder.find() expects a FluidCube instance.")

        report = _build_reporter(progress)

        if cube.is_chunked:
            return self._find_chunked(cube, output=output, progress=progress)
        if output is not None:
            raise ValueError(
                "output= is only used for chunked FluidCube inputs; "
                "use save_result_hdf5() for in-memory results."
            )
        return self._find_in_memory(cube, report=report)

    def _find_in_memory(
        self,
        cube: FluidCube,
        *,
        report: ProgressCallback,
    ) -> ShockFinderResult:
        cube_pass = analyze_cube_pass(cube, self.config, report=report)

        if self.config.reduce_to_centers:
            report("[8/9] Reducing to center cells")
            shock_mask, _ = reduce_to_centers(
                cube_pass.full_shock_mask,
                cube_pass.derived.div_v,
                cube_pass.mach_fields,
                center_score_mode=self.config.center_score,
            )
            report("[9/9] Building summary")
        else:
            shock_mask = cube_pass.full_shock_mask.copy()
            report("[8/8] Building summary")

        summary = make_summary(
            cube=cube,
            shock_zone_mask=cube_pass.shock_zone_mask,
            full_shock_mask=cube_pass.full_shock_mask,
            shock_mask=shock_mask,
            mach_fields=cube_pass.mach_fields,
            config=self.config,
        )
        return ShockFinderResult(
            shock_mask=shock_mask,
            shock_zone_mask=cube_pass.shock_zone_mask,
            mach_temperature=cube_pass.mach_fields.mach_temperature,
            mach_pressure=cube_pass.mach_fields.mach_pressure,
            compression=cube_pass.derived.compression,
            div_v=cube_pass.derived.div_v,
            temperature=cube_pass.derived.temperature,
            entropy=cube_pass.derived.entropy,
            temperature_jump=cube_pass.jumps.temperature_jump,
            pressure_jump=cube_pass.jumps.pressure_jump,
            density_jump=cube_pass.jumps.density_jump,
            normal_x=cube_pass.normals.normal_x,
            normal_y=cube_pass.normals.normal_y,
            normal_z=cube_pass.normals.normal_z,
            summary=summary,
            full_shock_mask=cube_pass.full_shock_mask,
        )

    def _find_chunked(
        self,
        cube: FluidCube,
        *,
        output: str | Path | None,
        progress: ProgressReporter,
    ) -> ChunkedShockFinderResult:
        if output is None:
            raise ValueError(
                "Chunked FluidCube requires output=... because result fields are written chunked on disk."
            )

        from .chunking import NpzChunkedInput, NpzChunkedOutput, run_chunked_shock_finder

        input_store = NpzChunkedInput.from_fluid_cube(cube)
        output_root = Path(output)
        output_store = NpzChunkedOutput(output_root, input_store.layout)
        summary = run_chunked_shock_finder(
            input_store,
            output_store,
            config=self.config,
            dx=cube.dx,
            dy=cube.dy,
            dz=cube.dz,
            gamma=cube.gamma,
            progress=progress,
        )
        return _build_chunked_result(output_root, input_store.layout, summary)


def analyze_cube_pass(
    cube: FluidCube,
    config: ShockFinderConfig,
    *,
    report: ProgressCallback | None = None,
) -> _FinderPass:
    """Run the local shock-finder physics on one cube."""

    total_steps = 9 if config.reduce_to_centers else 8
    if report is not None:
        report(f"[1/{total_steps}] Computing derived fields")
    derived = compute_derived_fields(cube)
    if report is not None:
        report(f"[2/{total_steps}] Computing gradients")
    gradients = compute_gradients(cube, derived.temperature, derived.entropy)
    if report is not None:
        report(f"[3/{total_steps}] Building shock-zone mask")
    shock_zone_mask = build_shock_zone_mask(derived, gradients, config)
    if report is not None:
        report(f"[4/{total_steps}] Computing shock normals")
    normals = compute_normals(gradients, config.normal_field)
    if report is not None:
        report(f"[5/{total_steps}] Sampling upstream and downstream jumps")
    jumps = sample_jumps(
        pressure=cube.pressure,
        temperature=derived.temperature,
        rho=cube.rho,
        nx=normals.normal_x,
        ny=normals.normal_y,
        nz=normals.normal_z,
        width=config.shock_width_cells,
        method=config.sampling_method,
        candidate_mask=shock_zone_mask,
    )
    if report is not None:
        report(f"[6/{total_steps}] Computing Mach fields")
    mach_fields = compute_mach_fields(
        jumps,
        cube.gamma,
        mach_max=config.mach_max,
        candidate_mask=shock_zone_mask,
    )
    if report is not None:
        report(f"[7/{total_steps}] Applying final shock filters")
    full_shock_mask = build_final_shock_mask(
        shock_zone_mask,
        jumps,
        mach_fields,
        config,
        cube.gamma,
    )
    return _FinderPass(
        derived=derived,
        normals=normals,
        shock_zone_mask=shock_zone_mask,
        jumps=jumps,
        mach_fields=mach_fields,
        full_shock_mask=full_shock_mask,
    )


def _build_reporter(progress: ProgressReporter) -> ProgressCallback:
    if progress is True or progress is None:
        return print
    if progress is False:
        return _noop_report
    return progress


def _noop_report(_: str) -> None:
    return None


def _build_chunked_result(
    output_root: Path,
    layout,
    summary: dict[str, object],
) -> ChunkedShockFinderResult:
    field_refs = {
        field_name: ChunkedFieldReference(path=output_root / field_name, layout=layout)
        for field_name in RESULT_FIELD_NAMES
    }
    return ChunkedShockFinderResult(
        output_root=output_root,
        shock_mask=field_refs["shock_mask"],
        shock_zone_mask=field_refs["shock_zone_mask"],
        mach_temperature=field_refs["mach_temperature"],
        mach_pressure=field_refs["mach_pressure"],
        compression=field_refs["compression"],
        div_v=field_refs["div_v"],
        temperature=field_refs["temperature"],
        entropy=field_refs["entropy"],
        temperature_jump=field_refs["temperature_jump"],
        pressure_jump=field_refs["pressure_jump"],
        density_jump=field_refs["density_jump"],
        normal_x=field_refs["normal_x"],
        normal_y=field_refs["normal_y"],
        normal_z=field_refs["normal_z"],
        summary=summary,
        full_shock_mask=field_refs["full_shock_mask"],
    )

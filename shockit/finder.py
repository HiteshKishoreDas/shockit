"""Main shock finding API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .chunked import iter_balanced_slices
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
from .result import ShockFinderResult
from .sampling import SampledJumps, sample_jumps
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
    ) -> ShockFinderResult:
        """Run the shock finder as a clear sequence of reviewable steps."""

        progress_callback: ProgressCallback | None
        if progress is True or progress is None:
            progress_callback = print
        elif progress is False:
            progress_callback = None
        else:
            progress_callback = progress

        def report(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)

        use_chunking = self._use_chunking(cube)
        if use_chunking:
            cube_pass = self._run_chunked_pass(cube, report)
        else:
            cube_pass = self._run_full_pass(cube, report)

        summary_step = 3 if use_chunking and self.config.reduce_to_centers else 2 if use_chunking else 9 if self.config.reduce_to_centers else 8
        reduction_step = 2 if use_chunking else 8

        if self.config.reduce_to_centers:
            if use_chunking:
                report(f"[{reduction_step}/{summary_step}] Reducing to center cells")
            else:
                report(f"[{reduction_step}/{summary_step}] Reducing to center cells")
            shock_mask, _ = reduce_to_centers(
                cube_pass.full_shock_mask,
                cube_pass.derived.div_v,
                cube_pass.mach_fields,
                center_score_mode=self.config.center_score,
            )
        else:
            shock_mask = cube_pass.full_shock_mask.copy()

        report(f"[{summary_step}/{summary_step}] Building summary")
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

    def _run_full_pass(self, cube: FluidCube, report: ProgressCallback) -> _FinderPass:
        total_steps = 9 if self.config.reduce_to_centers else 8
        report(f"[1/{total_steps}] Computing derived fields")
        derived = compute_derived_fields(cube)
        report(f"[2/{total_steps}] Computing gradients")
        gradients = compute_gradients(cube, derived.temperature, derived.entropy)
        report(f"[3/{total_steps}] Building shock-zone mask")
        shock_zone_mask = build_shock_zone_mask(derived, gradients, self.config)
        report(f"[4/{total_steps}] Computing shock normals")
        normals = compute_normals(gradients, self.config.normal_field)
        report(f"[5/{total_steps}] Sampling upstream and downstream jumps")
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
        report(f"[6/{total_steps}] Computing Mach fields")
        mach_fields = compute_mach_fields(
            jumps,
            cube.gamma,
            mach_max=self.config.mach_max,
            candidate_mask=shock_zone_mask,
        )
        report(f"[7/{total_steps}] Applying final shock filters")
        full_shock_mask = build_final_shock_mask(shock_zone_mask, jumps, mach_fields, self.config, cube.gamma)
        return _FinderPass(
            derived=derived,
            normals=normals,
            shock_zone_mask=shock_zone_mask,
            jumps=jumps,
            mach_fields=mach_fields,
            full_shock_mask=full_shock_mask,
        )

    def _run_chunked_pass(self, cube: FluidCube, report: ProgressCallback) -> _FinderPass:
        chunk_size = self.config.chunk_size
        if chunk_size is None:
            raise ValueError("chunked pass requires chunk_size.")

        halo = max(1, self.config.shock_width_cells)
        shape = cube.rho.shape
        chunk_slices = list(iter_balanced_slices(shape, chunk_size))
        total_chunks = len(chunk_slices)
        summary_step = 3 if self.config.reduce_to_centers else 2
        report(f"[1/{summary_step}] Processing {total_chunks} haloed chunks of up to {chunk_size}^3 cells")

        derived = DerivedFields(
            temperature=np.empty(shape, dtype=float),
            entropy=np.empty(shape, dtype=float),
            div_v=np.empty(shape, dtype=float),
            compression=np.empty(shape, dtype=float),
        )
        normals = NormalFields(
            normal_x=np.empty(shape, dtype=float),
            normal_y=np.empty(shape, dtype=float),
            normal_z=np.empty(shape, dtype=float),
        )
        shock_zone_mask = np.empty(shape, dtype=bool)
        jumps = SampledJumps(
            pressure_jump=np.empty(shape, dtype=float),
            temperature_jump=np.empty(shape, dtype=float),
            density_jump=np.empty(shape, dtype=float),
            pressure_up=np.empty(shape, dtype=float),
            pressure_down=np.empty(shape, dtype=float),
            temperature_up=np.empty(shape, dtype=float),
            temperature_down=np.empty(shape, dtype=float),
            density_up=np.empty(shape, dtype=float),
            density_down=np.empty(shape, dtype=float),
        )
        mach_fields = MachFields(
            mach_pressure=np.empty(shape, dtype=float),
            mach_temperature=np.empty(shape, dtype=float),
        )
        full_shock_mask = np.empty(shape, dtype=bool)

        for chunk_index, core_slice in enumerate(chunk_slices, start=1):
            report(f"[1/{summary_step}] Chunk {chunk_index}/{total_chunks} {core_slice}")
            local_cube = _extract_chunk_cube(cube, core_slice, halo)
            local_pass = analyze_cube_pass(local_cube, self.config)
            core_crop = tuple(slice(halo, halo + (sl.stop - sl.start)) for sl in core_slice)
            _assign_chunk(derived.temperature, core_slice, local_pass.derived.temperature[core_crop])
            _assign_chunk(derived.entropy, core_slice, local_pass.derived.entropy[core_crop])
            _assign_chunk(derived.div_v, core_slice, local_pass.derived.div_v[core_crop])
            _assign_chunk(derived.compression, core_slice, local_pass.derived.compression[core_crop])
            _assign_chunk(normals.normal_x, core_slice, local_pass.normals.normal_x[core_crop])
            _assign_chunk(normals.normal_y, core_slice, local_pass.normals.normal_y[core_crop])
            _assign_chunk(normals.normal_z, core_slice, local_pass.normals.normal_z[core_crop])
            _assign_chunk(shock_zone_mask, core_slice, local_pass.shock_zone_mask[core_crop])
            _assign_chunk(jumps.pressure_jump, core_slice, local_pass.jumps.pressure_jump[core_crop])
            _assign_chunk(jumps.temperature_jump, core_slice, local_pass.jumps.temperature_jump[core_crop])
            _assign_chunk(jumps.density_jump, core_slice, local_pass.jumps.density_jump[core_crop])
            _assign_chunk(jumps.pressure_up, core_slice, local_pass.jumps.pressure_up[core_crop])
            _assign_chunk(jumps.pressure_down, core_slice, local_pass.jumps.pressure_down[core_crop])
            _assign_chunk(jumps.temperature_up, core_slice, local_pass.jumps.temperature_up[core_crop])
            _assign_chunk(jumps.temperature_down, core_slice, local_pass.jumps.temperature_down[core_crop])
            _assign_chunk(jumps.density_up, core_slice, local_pass.jumps.density_up[core_crop])
            _assign_chunk(jumps.density_down, core_slice, local_pass.jumps.density_down[core_crop])
            _assign_chunk(mach_fields.mach_pressure, core_slice, local_pass.mach_fields.mach_pressure[core_crop])
            _assign_chunk(mach_fields.mach_temperature, core_slice, local_pass.mach_fields.mach_temperature[core_crop])
            _assign_chunk(full_shock_mask, core_slice, local_pass.full_shock_mask[core_crop])

        return _FinderPass(
            derived=derived,
            normals=normals,
            shock_zone_mask=shock_zone_mask,
            jumps=jumps,
            mach_fields=mach_fields,
            full_shock_mask=full_shock_mask,
        )

    def _use_chunking(self, cube: FluidCube) -> bool:
        chunk_size = self.config.chunk_size
        return chunk_size is not None and any(axis_size > chunk_size for axis_size in cube.rho.shape)


def analyze_cube_pass(cube: FluidCube, config: ShockFinderConfig) -> _FinderPass:
    """Run the local shock-finder physics on one cube."""

    derived = compute_derived_fields(cube)
    gradients = compute_gradients(cube, derived.temperature, derived.entropy)
    shock_zone_mask = build_shock_zone_mask(derived, gradients, config)
    normals = compute_normals(gradients, config.normal_field)
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
    mach_fields = compute_mach_fields(
        jumps,
        cube.gamma,
        mach_max=config.mach_max,
        candidate_mask=shock_zone_mask,
    )
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


def _extract_chunk_cube(cube: FluidCube, core_slice: tuple[slice, slice, slice], halo: int) -> FluidCube:
    return FluidCube(
        rho=_extract_periodic_subcube(cube.rho, core_slice, halo),
        pressure=_extract_periodic_subcube(cube.pressure, core_slice, halo),
        vx=_extract_periodic_subcube(cube.vx, core_slice, halo),
        vy=_extract_periodic_subcube(cube.vy, core_slice, halo),
        vz=_extract_periodic_subcube(cube.vz, core_slice, halo),
        dx=cube.dx,
        dy=cube.dy,
        dz=cube.dz,
        gamma=cube.gamma,
        metadata=cube.metadata,
    )


def _extract_periodic_subcube(field: np.ndarray, core_slice: tuple[slice, slice, slice], halo: int) -> np.ndarray:
    chunk = field
    for axis, axis_slice in enumerate(core_slice):
        indices = np.arange(axis_slice.start - halo, axis_slice.stop + halo)
        chunk = np.take(chunk, indices, axis=axis, mode="wrap")
    return np.asarray(chunk, dtype=float)


def _assign_chunk(target: np.ndarray, core_slice: tuple[slice, slice, slice], values: np.ndarray) -> None:
    target[core_slice] = values

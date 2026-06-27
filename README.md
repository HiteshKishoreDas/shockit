# ShocKit

A Python toolkit for shock detection and characterization in simulation data.

`shockit` is a standalone Python package for offline shock finding on
already-extracted uniform 3D NumPy cubes of primitive hydrodynamic variables.
The core algorithm is independent of AthenaK, yt, AMR, and HDF5. HDF5 support
is provided through a small adapter layer.

The package targets ideal-gas hydrodynamics on uniform Cartesian grids with
periodic finite-difference operators in version 1.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Minimal array example

```python
import numpy as np
from shockit import FluidCube, ShockFinder, ShockFinderConfig

cube = FluidCube(
    rho=rho,
    pressure=pressure,
    vx=vx,
    vy=vy,
    vz=vz,
    dx=dx,
    dy=dy,
    dz=dz,
    gamma=5.0 / 3.0,
)

config = ShockFinderConfig(min_mach=1.2, shock_width_cells=1)
result = ShockFinder(config).find(cube)

shock_mask = result.shock_mask
```

## HDF5 CLI example

```bash
shockit-find snapshot.h5 \
  --rho-field rho \
  --pressure-field press \
  --vx-field vel1 \
  --vy-field vel2 \
  --vz-field vel3 \
  --gamma 1.6666666667 \
  --min-mach 1.2 \
  --output shocks.h5
```

## Chunked NPZ example

```python
from shockit import ShockFinderConfig, run_npz_shock_finder

summary = run_npz_shock_finder(
    "sim_data_2",
    ShockFinderConfig(
        reduce_to_centers=True,
        min_mach=1.1,
        sampling_method="nearest_axis",
    ),
)
```

`run_npz_shock_finder()` is the unified entry point for both plain and chunked
`.npz` layouts. It auto-detects whether the directory contains:

- plain `rho.npz` / `prs.npz` / `v1.npz` / `v2.npz` / `v3.npz`, or
- a `chunked_npz/` tree of chunk directories.

For chunked inputs, it reads neighboring chunk files to assemble haloed local
cubes, runs the same shock physics on each local region, writes chunked shock
outputs to `shock_chunked_npz/`, and, when requested, runs a separate second
pass to reduce centers across chunk boundaries.

## How the finder works

The implementation follows a Skillman-style workflow:

1. Compute derived thermodynamic proxies:
   - `temperature = pressure / rho`
   - `entropy = log(pressure / rho**gamma)`
2. Compute periodic centered gradients and velocity divergence.
3. Mark preliminary shock-zone cells where:
   - `div_v < div_v_threshold`
   - `|grad_T| > grad_T_min`
   - `grad_T . grad_S > 0` when enabled
   - `grad_T . grad_rho > 0` when enabled
4. Build a shock normal from either the temperature or pressure gradient.
5. Sample upstream and downstream states with `sampling_method`:
   - `nearest_axis` uses the dominant grid axis and periodic whole-cube array rolls
   - `trilinear` uses periodic trilinear interpolation on candidate cells only
6. Require consistent pressure, temperature, and density jumps when enabled.
7. Reject sampled jumps that violate the ideal-gas density-jump limit or that
   fail `temperature_jump ~= pressure_jump / density_jump`.
8. Optionally reject sampled cells whose upstream pressure, temperature, or
   density falls below configured floors.
9. Estimate Mach number from Rankine-Hugoniot jump relations.
10. Keep cells that satisfy the zone criteria, jump criteria, and minimum Mach.
11. Optionally reduce connected shock regions to representative center cells.

Mask semantics:

- `shock_zone_mask`: local compression and gradient-alignment candidate zone
- `full_shock_mask`: unreduced shock cells that pass jump and Mach filtering
- `shock_mask`: final public mask and may be center-reduced

`nearest_axis` is simple and memory-predictable, but it is not candidate-only.
`trilinear` samples candidate cells only and is useful when oblique-shock
normal sampling is specifically needed. For large production snapshots, default
to `nearest_axis` unless you need that oblique sampling behavior.

When loading HDF5 snapshots through the CLI, compute-time chunking now defaults
to the source dataset chunk shape when all primitive fields share one. Pass
`--chunk-size N` to override that, or `--chunk-size 0` to force an unchunked
full-cube pass.

## Mach estimates

The package returns two Mach estimates:

- `mach_pressure`, from the analytic inversion of `P2 / P1`
- `mach_temperature`, from a robust bisection inversion of `T2 / T1`

For broadened numerical shocks, the recovered Mach depends on the sampled
upstream/downstream cells and may be biased low or high.

## Public API

- `FluidCube`: validated container for primitive fields and grid spacing
- `ShockFinderConfig`: knobs for thresholds, jump requirements, center scoring,
  and sampling mode
- `ShockFinder`: main finder class
- `run_npz_shock_finder()`: one-call runner for both plain and chunked `.npz`
  simulation directories
- `run_chunked_npz_shock_finder()`: streamed shock finding on chunked `.npz`
  cube directories with chunked outputs
- `load_fluid_cube_from_hdf5()`: adapter for explicit paths, top-level aliases,
  and unique nested dataset basenames
- `join_chunked_field()`, `join_all_fields()`: rebuild full arrays from chunked
  `.npz` directories for plotting or inspection
- `save_result_hdf5()`: save masks, jumps, Mach fields, normals, and summary

## Manual review

- [docs/review_map.md](docs/review_map.md): file-by-file map for manual review

## Test suite

The included tests cover:

- package import and CLI parser smoke tests
- Mach inversion accuracy
- periodic gradients and divergence
- planar shock detection
- oblique shock detection
- contact discontinuity rejection
- a Sod-like fixture
- HDF5 field loading, including nested dataset paths
- configuration and data validation
- repository hygiene checks for stale names and absolute links

Run them with:

```bash
python -m pytest
```

To save viewable plot artifacts from the plotting-aware tests:

```bash
python -m pytest --plot-tests
```

PNG files are written under `test_artifacts/plots/` by default. You can choose
another directory with `--plot-dir`.

## Limitations

- uniform grid only
- ideal gas only
- hydrodynamics only
- periodic finite-difference boundaries only in v1
- no AMR support
- no SPH support
- no yt frontend
- no distributed execution; local haloed chunk processing is available
- shock broadening affects Mach estimates
- false positives are still possible in compressive turbulent regions
- contact rejection is helpful but not perfect

## Examples

- [examples/run_from_arrays.py](examples/run_from_arrays.py)
- [examples/run_from_hdf5.py](examples/run_from_hdf5.py)
- [examples/make_synthetic_planar_shock.py](examples/make_synthetic_planar_shock.py)

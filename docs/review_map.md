# Manual review map

## User-facing API

1. `fields.py`
   - `FluidCube` accepts either in-memory arrays or chunked field paths
   - invariants: storage mode is all-array or all-path; chunked validation is metadata-only

2. `finder.py`
   - `ShockFinder.find(cube, output=None)` dispatches on `cube.storage_mode`
   - invariants: in-memory cubes return `ShockFinderResult`; chunked cubes require `output=...`

3. `result.py`
   - in-memory results use arrays; chunked results expose lightweight field handles
   - invariants: result attribute names stay parallel across storage modes

## Core algorithm path

1. `gradients.py`
   - periodic finite-difference operators
   - invariants: centered second-order stencil, periodic wrapping

2. `derived.py`
   - temperature/entropy proxies, gradients, normals, Mach fields
   - invariants: no physics beyond ideal-gas proxies; pressure Mach relation analytic

3. `sampling.py`
   - upstream/downstream sampling
   - invariants: downstream is higher-pressure side; invalid ratios become NaN

4. `masks.py`
   - shock-zone criteria, final mask, center reduction
   - invariants: `shock_mask <= full_shock_mask <= shock_zone_mask`

5. `finder.py`
   - shared in-memory algorithm path and dispatch entrypoint
   - `analyze_cube_pass()` should stay chunk-agnostic and reviewable as the algorithm recipe

## Chunked workflow review path

1. `chunking/layout.py`
   - chunk specs and regular-grid layout inference
   - invariants: chunk coverage is complete and non-overlapping

2. `chunking/reader.py`
   - chunked field reader protocols and `.npz` input implementation
   - invariants: halo reads are periodic and layout-consistent across fields

3. `chunking/runner.py`
   - chunked orchestration over haloed local cubes
   - invariants: uses the same `analyze_cube_pass()` physics as the in-memory path
   - converts chunked `FluidCube` storage into local in-memory chunks plus chunked outputs

4. `chunking/centers.py`
   - local labels, boundary merging, and global center reduction
   - invariants: no full 3D field reconstruction during reduction

5. `chunking/writer.py`
   - chunked field writer protocol and `.npz` output implementation
   - invariants: core-chunk metadata is preserved exactly

## I/O adapter review path

1. `io_hdf5.py`
   - optional HDF5 adapter
   - exact paths preferred; ambiguous basename matches raise

2. `output.py`
   - result writer
   - mask datasets include semantic descriptions

3. `chunking/reader.py` and `chunking/writer.py`
   - first chunk-store adapters (`NpzChunkedInput` / `NpzChunkedOutput`)
   - should stay format-specific and avoid leaking storage concerns into the physics path

4. `summary.py`
   - summary statistics
   - distinguishes full vs center-reduced masks

## Test review path

1. `docs/test_suite.md`
   - complete inventory of test modules and how to run them
   - includes the opt-in chunked stress-test controls

2. `tests/test_chunked_finder.py` and `tests/test_center_reduction.py`
   - main regression path for chunk-aware equivalence and global center reduction

3. `tests/test_validation.py` and `tests/test_package_layout.py`
   - API-surface and configuration guardrails

4. `tests/test_planar_shock.py`, `tests/test_oblique_shock.py`, `tests/test_sod.py`, and `tests/test_contact_rejection.py`
   - physics-behavior smoke tests on representative fixtures

# Manual review map

## Core algorithm path

1. `fields.py`
   - validates primitive fields
   - invariants: 3D arrays, matching shapes, finite values, positive rho/pressure

2. `gradients.py`
   - periodic finite-difference operators
   - invariants: centered second-order stencil, periodic wrapping

3. `derived.py`
   - temperature/entropy proxies, gradients, normals, Mach fields
   - invariants: no physics beyond ideal-gas proxies; pressure Mach relation analytic

4. `sampling.py`
   - upstream/downstream sampling
   - invariants: downstream is higher-pressure side; invalid ratios become NaN

5. `masks.py`
   - shock-zone criteria, final mask, center reduction
   - invariants: `shock_mask <= full_shock_mask <= shock_zone_mask`

6. `finder.py`
   - orchestration only for the in-memory algorithm
   - should read like the algorithm recipe

## Chunked workflow review path

1. `chunking/layout.py`
   - chunk specs and regular-grid layout inference
   - invariants: chunk coverage is complete and non-overlapping

2. `chunking/reader.py`
   - chunked field reader protocols and `.npz` input implementation
   - invariants: halo reads are periodic and layout-consistent across fields

3. `chunking/runner.py`
   - explicit chunked orchestration
   - invariants: uses the same `analyze_cube_pass()` physics as `ShockFinder.find()`
   - chunking is selected by the workflow input/output stores, not by core finder config

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

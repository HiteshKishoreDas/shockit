# Chunked center reduction

This document describes the optional/manual chunked center-reduction pass.
The default unified chunked workflow does not run this step because it can be
too memory-hungry on large datasets.

When run manually, the reducer turns unreduced shock cells into representative
center cells without reconstructing every output field as one full 3D array.

## Overview

1. Each output chunk labels its local connected components in
   `full_shock_mask`.
2. Neighboring chunk faces are compared and matching labels are merged with a
   union-find structure.
3. Periodic chunk faces are also compared so components can merge across domain
   boundaries.
4. Each merged component chooses one global center cell using the configured
   `center_score` mode.
5. The final `shock_mask` is written back sparsely as chunk-core boolean masks.

## Local labels

`shockit/chunking/centers.py` first runs `scipy.ndimage.label` on each chunk’s
`full_shock_mask`. For each local label it records:

- the chunk grid index
- the local best candidate
- the corresponding global flat index
- the local best Mach values

These records let the chunked workflow reason about connected components without
loading unrelated chunks.

## Boundary merging

Chunk faces are compared along the positive direction of each axis. When a
nonzero label touches a nonzero label on the neighboring face, those local
labels are unioned.

This step runs both for interior neighbors and periodic wrap neighbors.

## Global center selection

For simple score modes (`compression`, `mach_pressure`,
`mach_temperature`), merged groups are assembled into a small bounding-box view
when possible so the chunked path matches the in-memory `reduce_to_centers()`
behavior exactly.

For merged groups that are only connected through periodic wrap, the reducer
falls back to sparse per-label comparison while preserving deterministic
selection.

For `combined`, the reducer normalizes compression and Mach over the merged
group and then selects the best global cell deterministically.

## Output

The reducer writes one sparse boolean `shock_mask` chunk per input/output
chunk. `full_shock_mask` stays unreduced, and `summary.json` records both
unreduced and center-reduced counts.

# Shockit Algorithm

`shockit` operates on uniform-grid primitive fields `rho`, `pressure`, `vx`,
`vy`, and `vz`.

## Derived thermodynamic proxies

- Temperature proxy: `T ~ pressure / rho`
- Entropy proxy: `S ~ log(pressure / rho**gamma)`
- Compression proxy: `compression = -div(v)`

These are proxy fields, not absolute thermodynamic units.

## Local shock-zone criteria

A cell enters `shock_zone_mask` when it is locally compressive and its
temperature gradient is non-trivial:

- `div(v) < div_v_threshold`
- `|grad(T)| > grad_T_min`
- optionally `grad(T) . grad(S) > 0`
- optionally `grad(T) . grad(rho) > 0`

This mask is only a candidate zone. It does not yet require a resolved shock
jump.

## Shock normal choice

The local shock normal is the normalized gradient of either:

- temperature proxy, or
- pressure

depending on `normal_field`.

## Upstream and downstream sampling

`shockit` samples states one or more cells away along the chosen normal.

- `nearest_axis`: step along the dominant normal axis using periodic array
  rolls over the whole cube
- `trilinear`: interpolate along the full normal using periodic trilinear
  sampling on candidate cells only, then scatter the results back into full
  arrays

For each candidate cell, the higher-pressure side is treated as downstream and
the lower-pressure side as upstream.

`nearest_axis` is simple and memory-predictable, but it is not candidate-only.
`trilinear` is candidate-only and better matched to oblique normals, but it is
more specialized.

## Jump filtering

Sampled jump ratios are:

- `pressure_jump = P_down / P_up`
- `temperature_jump = T_down / T_up`
- `density_jump = rho_down / rho_up`

`full_shock_mask` keeps only `shock_zone_mask` cells that satisfy the enabled
jump filters and the minimum Mach threshold.

Additional physical rejection is always applied before a cell enters
`full_shock_mask`:

- `density_jump` must not exceed the ideal-gas normal-shock limit
  `(gamma + 1) / (gamma - 1)`
- the sampled triplet must be thermodynamically consistent, with
  `temperature_jump ~= pressure_jump / density_jump`

Optional upstream-state floors can also be enabled through
`ShockFinderConfig` to reject sampled cells whose upstream pressure,
temperature, or density is too close to vacuum for a trustworthy jump ratio.

## Mach reconstruction

Two Mach estimates are computed:

- `mach_pressure` from analytic inversion of the pressure jump
- `mach_temperature` from inversion of the temperature jump

Cells pass the Mach filter if either estimate is at least `min_mach`.
Mach fields are only meaningful inside `shock_zone_mask`/`full_shock_mask`;
outside candidate regions they may be `NaN`.

## Mask semantics

- `shock_zone_mask`: local compression and gradient-alignment candidate zone
- `full_shock_mask`: unreduced candidate cells that also pass jump and Mach
  filtering
- `shock_mask`: final public mask; may be center-reduced to one cell per
  connected component

When `reduce_to_centers=True`, `shock_mask` is intended for cataloging rather
than geometric thickness.

## Chunking note

Chunking is a workflow implementation detail, not a separate physics
algorithm. The explicit chunked runner reads haloed primitive fields, applies
the same local `analyze_cube_pass()` physics as the in-memory finder, and then
reduces centers globally across chunk boundaries. Equivalent chunked and
non-chunked inputs should agree within normal floating-point tolerance.

## Limitations

- Uniform Cartesian grids only
- Ideal-gas hydrodynamics only
- Periodic finite-difference gradients
- Shock broadening can bias reconstructed Mach numbers
- Strong compression alone is not sufficient for a shock; the alignment and
  jump filters reduce, but do not eliminate, false positives
- Trilinear sampling is intended for small or moderate candidate sets, not for
  blindly sampling every cell in very large cubes

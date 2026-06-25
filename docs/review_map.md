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
   - orchestration only
   - should read like the algorithm recipe

7. `io_hdf5.py`
   - optional HDF5 adapter
   - exact paths preferred; ambiguous basename matches raise

8. `output.py`
   - result writer
   - mask datasets include semantic descriptions

9. `summary.py`
   - summary statistics
   - distinguishes full vs center-reduced masks

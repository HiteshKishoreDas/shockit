# Test suite map

This document describes the purpose of every test module under `tests/` and
how to run the main test modes.

For command-oriented test-running notes, see [docs/testing.md](testing.md).

## Running tests

Run the default suite with the repo-local virtualenv:

```bash
./.venv/bin/python -m pytest -q
```

Some tests can save plot artifacts for manual inspection:

```bash
./.venv/bin/python -m pytest --plot-tests
```

PNG files are written under `test_artifacts/plots/` by default. Override that
directory with `--plot-dir`.

## Stress test mode

`tests/test_chunked_stress.py` is opt-in because it writes and processes a
large synthetic chunked cube.

Enable it with:

```bash
SHOCKIT_RUN_STRESS_TESTS=1 ./.venv/bin/python -m pytest tests/test_chunked_stress.py -q
```

Optional knobs:

- `SHOCKIT_STRESS_SHAPE`: one integer or three comma-separated integers.
  Default is `256,256,256`.
- `SHOCKIT_STRESS_CHUNK_SHAPE`: one integer or three comma-separated integers.
  Default is `128,128,128`.
- `SHOCKIT_STRESS_SEED`: deterministic seed for the synthetic chunked cube.

For a true large-scale chunking run, set `SHOCKIT_STRESS_SHAPE=2048`.

## Test modules

### Suite support

- `tests/conftest.py`: shared pytest options and the `test_plotter` fixture for
  plot-producing tests.
- `tests/helpers.py`: reusable synthetic planar, oblique, and periodic-edge
  shock fixtures.

### Core physics and numerics

- `tests/test_physical_behavior.py`: deterministic no-shock and rejected-shock
  fixtures such as uniform flow, pressure-without-compression, and compression
  without thermodynamic jumps.
- `tests/test_mach.py`: validates Mach inversion from pressure and temperature
  jumps and checks invalid jump handling.
- `tests/test_gradients.py`: verifies periodic derivatives and divergence
  against analytic reference functions.
- `tests/test_derived.py`: checks candidate-only handling in derived Mach-field
  computation.
- `tests/test_sampling.py`: covers `nearest_axis` and `trilinear` sampling,
  candidate-only trilinear work, and invalid upstream-state handling.
- `tests/test_masks.py`: verifies mask nesting, unreduced-mask behavior, jump
  consistency filtering, and optional upstream floors.
- `tests/test_summary.py`: checks summary semantics for center-reduced results.
- `tests/test_progress.py`: verifies progress-step reporting for default,
  reduced, and silent finder execution.

### Shock-detection behavior

- `tests/test_planar_shock.py`: checks planar shock localization and recovered
  Mach behavior.
- `tests/test_oblique_shock.py`: checks oblique-shock detection and usable
  normal alignment.
- `tests/test_contact_rejection.py`: checks that a contact discontinuity is not
  promoted to a shock.
- `tests/test_sod.py`: exercises a Sod-like setup and checks that the shock
  dominates over the contact.

### Chunking and chunk-aware reduction

- `tests/test_chunk_layout_validation.py`: validates that chunk extents start at
  zero, end at the full shape, and tile each axis without gaps or overlaps.
- `tests/test_chunked_reader.py`: verifies periodic halo reads for interior,
  boundary, corner, and larger-halo chunk requests.
- `tests/test_chunking.py`: verifies required halo width and balanced chunk
  layout generation.
- `tests/test_chunk_join.py`: checks reconstruction of a full field from chunk
  files.
- `tests/test_center_reduction.py`: tests center-score rules plus chunk-boundary
  and periodic-boundary component merging during reduction.
- `tests/test_chunked_finder.py`: covers unified in-memory/chunked dispatch,
  chunked output requirements, and chunked-vs-full-cube equivalence for planar,
  oblique, and periodic-edge fixtures.
- `tests/test_chunked_stress.py`: opt-in large synthetic chunked pipeline test
  that writes a deterministic smooth-noise cube directly as chunk files and
  runs the chunked finder end to end.

### I/O adapters and outputs

- `tests/test_hdf5_adapter.py`: validates HDF5 field loading, nested dataset
  paths, basename fallback rules, ambiguity errors, and source chunk metadata.
- `tests/test_output.py`: checks that written HDF5 masks include semantic
  descriptions.
- `tests/test_npz_pipeline.py`: tests plain-vs-chunked `.npz` layout detection
  and compatibility with the direct finder workflows.

### Package surface, validation, and docs

- `tests/test_finder_api.py`: covers `ShockFinder.find()` dispatch, missing
  chunked output, in-memory `output=` rejection, and bad-input type errors.
- `tests/test_result_interface.py`: checks interface parity between in-memory
  and chunked result objects.
- `tests/test_validation.py`: validates configuration errors, bad array inputs,
  chunked `FluidCube` mode detection, and chunked layout consistency checks.
- `tests/test_package_layout.py`: checks import smoke, explicit package exports,
  CLI parser behavior, and removed chunk-size options.
- `tests/test_repo_hygiene.py`: enforces repository hygiene constraints such as
  no stale package names, no absolute user paths, and expected finder/readme
  wording.
- `tests/test_review_docs.py`: checks that the review-map document exists and
  is linked from the README.

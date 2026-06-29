# Testing

## Default suite

Run the normal fast suite from the repo root with the repo-local virtualenv:

```bash
./.venv/bin/python -m pytest -q
```

This is the default correctness check for normal development.

## Focused runs

Run one module or one test when iterating on a narrow area:

```bash
./.venv/bin/python -m pytest tests/test_chunked_reader.py -q
./.venv/bin/python -m pytest tests/test_finder_api.py::test_find_rejects_output_for_in_memory_cube -q
```

## Plot artifacts

Some tests can save PNG artifacts for manual inspection:

```bash
./.venv/bin/python -m pytest --plot-tests
```

Use `--plot-dir` to choose a different output directory.

## Stress tests

The normal suite stays fast by keeping large chunked stress tests opt-in.

Run only stress tests:

```bash
SHOCKIT_RUN_STRESS_TESTS=1 ./.venv/bin/python -m pytest -m stress -q
```

Run the chunked stress module directly:

```bash
SHOCKIT_RUN_STRESS_TESTS=1 ./.venv/bin/python -m pytest tests/test_chunked_stress.py -q
```

Optional environment knobs:

- `SHOCKIT_STRESS_SHAPE`: one integer or three comma-separated integers
- `SHOCKIT_STRESS_CHUNK_SHAPE`: one integer or three comma-separated integers
- `SHOCKIT_STRESS_SEED`: deterministic seed for the synthetic cube

Example large run:

```bash
SHOCKIT_RUN_STRESS_TESTS=1 \
SHOCKIT_STRESS_SHAPE=2048 \
SHOCKIT_STRESS_CHUNK_SHAPE=128 \
./.venv/bin/python -m pytest tests/test_chunked_stress.py -q
```

## Test map

For a module-by-module description of the suite, see
[docs/test_suite.md](test_suite.md).

from __future__ import annotations

from pathlib import Path

import numpy as np

from shockit import FluidCube, ShockFinder, ShockFinderConfig
from shockit.chunking import save_chunked_field
from shockit.result import ChunkedShockFinderResult, RESULT_FIELD_NAMES, ShockFinderResult
from shockit.storage import ChunkedFieldReference
from tests.helpers import make_planar_shock_cube

FIELD_NAMES = (*RESULT_FIELD_NAMES, "summary")


def _chunked_cube(root: Path) -> FluidCube:
    cube = make_planar_shock_cube(mach=2.0, n=12)
    save_chunked_field("rho", cube.rho, root, target_chunk_size=(5, 7, 4))
    save_chunked_field("prs", cube.pressure, root, target_chunk_size=(5, 7, 4))
    save_chunked_field("v1", cube.vx, root, target_chunk_size=(5, 7, 4))
    save_chunked_field("v2", cube.vy, root, target_chunk_size=(5, 7, 4))
    save_chunked_field("v3", cube.vz, root, target_chunk_size=(5, 7, 4))
    return FluidCube(
        rho=root / "rho",
        pressure=root / "prs",
        vx=root / "v1",
        vy=root / "v2",
        vz=root / "v3",
    )


def test_in_memory_and_chunked_results_expose_parallel_field_names(tmp_path: Path) -> None:
    full_result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(
        make_planar_shock_cube(mach=2.0, n=12),
        progress=False,
    )
    chunked_result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(
        _chunked_cube(tmp_path / "input"),
        output=tmp_path / "output",
        progress=False,
    )

    assert isinstance(full_result, ShockFinderResult)
    assert isinstance(chunked_result, ChunkedShockFinderResult)
    for field_name in FIELD_NAMES:
        assert hasattr(full_result, field_name)
        assert hasattr(chunked_result, field_name)


def test_chunked_result_fields_expose_reference_interface(tmp_path: Path) -> None:
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=False, min_mach=1.1)).find(
        _chunked_cube(tmp_path / "input"),
        output=tmp_path / "output",
        progress=False,
    )

    assert isinstance(result, ChunkedShockFinderResult)
    for field_name in FIELD_NAMES:
        if field_name == "summary":
            continue
        field_value = getattr(result, field_name)
        assert isinstance(field_value, ChunkedFieldReference)
        assert field_value.path.exists()
        assert field_value.shape == (12, 12, 12)
        loaded = field_value.load()
        assert loaded.shape == field_value.shape
        iterated = list(field_value.iter_chunks())
        assert len(iterated) > 0
        assert sum(np.prod(values.shape) for _, values in iterated) == np.prod(field_value.shape)

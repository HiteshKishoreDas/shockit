from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shockit.chunking.layout import load_chunk_layout


def _write_chunk(
    field_dir: Path,
    *,
    filename: str,
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> None:
    shape = tuple(stop[axis] - start[axis] for axis in range(3))
    np.savez(
        field_dir / filename,
        values=np.zeros(shape, dtype=float),
        start=np.array(start, dtype=np.int64),
        stop=np.array(stop, dtype=np.int64),
        full_shape=np.array(full_shape, dtype=np.int64),
    )


def test_load_chunk_layout_accepts_exact_regular_tiling(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(4, 4, 4), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0001.npz", start=(0, 0, 4), stop=(4, 4, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0002.npz", start=(0, 4, 0), stop=(4, 8, 4), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0003.npz", start=(0, 4, 4), stop=(4, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0004.npz", start=(4, 0, 0), stop=(8, 4, 4), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0005.npz", start=(4, 0, 4), stop=(8, 4, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0006.npz", start=(4, 4, 0), stop=(8, 8, 4), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0007.npz", start=(4, 4, 4), stop=(8, 8, 8), full_shape=(8, 8, 8))

    layout = load_chunk_layout(field_dir)

    assert layout.shape == (8, 8, 8)
    assert layout.axis_slices[0][0] == slice(0, 4)
    assert layout.axis_slices[0][-1] == slice(4, 8)


def test_load_chunk_layout_rejects_gap_in_axis_coverage(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(4, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0001.npz", start=(5, 0, 0), stop=(8, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="gap on axis 0"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_overlap_in_axis_coverage(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(5, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0001.npz", start=(4, 0, 0), stop=(8, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="overlap on axis 0"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_axis_that_does_not_start_at_zero(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(1, 0, 0), stop=(8, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="starts at 1 instead of 0"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_axis_that_does_not_end_at_full_shape(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(7, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="ends at 7 instead of 8"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_zero_width_chunk(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(0, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="Invalid chunk extent"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_negative_width_chunk(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    np.savez(
        field_dir / "chunk_0000.npz",
        values=np.zeros((1, 8, 8), dtype=float),
        start=np.array([4, 0, 0], dtype=np.int64),
        stop=np.array([3, 8, 8], dtype=np.int64),
        full_shape=np.array([8, 8, 8], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Invalid chunk extent"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_duplicate_chunk_extent(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(4, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0001.npz", start=(0, 0, 0), stop=(4, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0002.npz", start=(4, 0, 0), stop=(8, 8, 8), full_shape=(8, 8, 8))

    with pytest.raises(ValueError, match="duplicate grid indices"):
        load_chunk_layout(field_dir)


def test_load_chunk_layout_rejects_inconsistent_full_shape(tmp_path: Path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()
    _write_chunk(field_dir, filename="chunk_0000.npz", start=(0, 0, 0), stop=(4, 8, 8), full_shape=(8, 8, 8))
    _write_chunk(field_dir, filename="chunk_0001.npz", start=(4, 0, 0), stop=(8, 8, 8), full_shape=(9, 8, 8))

    with pytest.raises(ValueError, match="Inconsistent full shapes"):
        load_chunk_layout(field_dir)

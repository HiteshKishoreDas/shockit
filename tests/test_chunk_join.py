from __future__ import annotations

from pathlib import Path

import numpy as np

from shockit.chunking import join_chunked_field


def test_join_chunked_field_reassembles_array(tmp_path) -> None:
    field_dir = tmp_path / "rho"
    field_dir.mkdir()

    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    chunk_specs = [
        ((slice(0, 1), slice(0, 3), slice(0, 2)), values[0:1, :, 0:2]),
        ((slice(0, 1), slice(0, 3), slice(2, 4)), values[0:1, :, 2:4]),
        ((slice(1, 2), slice(0, 3), slice(0, 2)), values[1:2, :, 0:2]),
        ((slice(1, 2), slice(0, 3), slice(2, 4)), values[1:2, :, 2:4]),
    ]

    for index, (chunk_slice, chunk_values) in enumerate(chunk_specs):
        np.savez(
            field_dir / f"chunk_{index:04d}.npz",
            values=chunk_values,
            start=np.array([chunk_slice[0].start, chunk_slice[1].start, chunk_slice[2].start], dtype=np.int64),
            stop=np.array([chunk_slice[0].stop, chunk_slice[1].stop, chunk_slice[2].stop], dtype=np.int64),
            full_shape=np.array(values.shape, dtype=np.int64),
        )

    rebuilt = join_chunked_field(field_dir)
    np.testing.assert_array_equal(rebuilt, values)

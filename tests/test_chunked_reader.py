from __future__ import annotations

import numpy as np

from shockit.chunking import NpzChunkedInput, save_chunked_field


def _periodic_window(values: np.ndarray, start: tuple[int, int, int], stop: tuple[int, int, int]) -> np.ndarray:
    result = values
    for axis in range(3):
        indices = np.arange(start[axis], stop[axis]) % values.shape[axis]
        result = np.take(result, indices, axis=axis)
    return result


def _linear_index_field(shape: tuple[int, int, int]) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


def test_read_with_halo_matches_direct_periodic_slice_for_interior_chunk(tmp_path) -> None:
    values = _linear_index_field((6, 6, 6))
    root = tmp_path / "chunked"
    save_chunked_field("rho", values, root, target_chunk_size=2)

    input_store = NpzChunkedInput(field_paths={"rho": root / "rho"})
    reader = input_store.field_reader("rho")
    spec = input_store.layout.spec_for_grid_index((1, 1, 1))

    actual = reader.read_with_halo(spec, halo=1)
    expected = _periodic_window(values, (1, 1, 1), (5, 5, 5))

    np.testing.assert_array_equal(actual, expected)


def test_read_with_halo_wraps_for_boundary_chunk(tmp_path) -> None:
    values = _linear_index_field((6, 6, 6))
    root = tmp_path / "chunked"
    save_chunked_field("rho", values, root, target_chunk_size=2)

    input_store = NpzChunkedInput(field_paths={"rho": root / "rho"})
    reader = input_store.field_reader("rho")
    spec = input_store.layout.spec_for_grid_index((0, 1, 1))

    actual = reader.read_with_halo(spec, halo=1)
    expected = _periodic_window(values, (-1, 1, 1), (3, 5, 5))

    np.testing.assert_array_equal(actual, expected)


def test_read_with_halo_wraps_for_corner_chunk(tmp_path) -> None:
    values = _linear_index_field((6, 6, 6))
    root = tmp_path / "chunked"
    save_chunked_field("rho", values, root, target_chunk_size=2)

    input_store = NpzChunkedInput(field_paths={"rho": root / "rho"})
    reader = input_store.field_reader("rho")
    spec = input_store.layout.spec_for_grid_index((0, 0, 0))

    actual = reader.read_with_halo(spec, halo=1)
    expected = _periodic_window(values, (-1, -1, -1), (3, 3, 3))

    np.testing.assert_array_equal(actual, expected)


def test_read_with_halo_supports_larger_halo_widths(tmp_path) -> None:
    values = _linear_index_field((8, 8, 8))
    root = tmp_path / "chunked"
    save_chunked_field("rho", values, root, target_chunk_size=4)

    input_store = NpzChunkedInput(field_paths={"rho": root / "rho"})
    reader = input_store.field_reader("rho")
    spec = input_store.layout.spec_for_grid_index((0, 0, 0))

    actual = reader.read_with_halo(spec, halo=3)
    expected = _periodic_window(values, (-3, -3, -3), (7, 7, 7))

    assert actual.shape == (10, 10, 10)
    np.testing.assert_array_equal(actual, expected)

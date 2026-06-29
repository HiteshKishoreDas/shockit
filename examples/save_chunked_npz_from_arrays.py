"""Save primitive arrays into a package-compatible chunked `.npz` layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shockit.chunking import save_chunked_field

FieldMap = dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=Path, required=True, help="Path to rho array (.npy or .npz).")
    parser.add_argument("--prs", type=Path, required=True, help="Path to pressure array (.npy or .npz).")
    parser.add_argument("--v1", type=Path, required=True, help="Path to x-velocity array (.npy or .npz).")
    parser.add_argument("--v2", type=Path, required=True, help="Path to y-velocity array (.npy or .npz).")
    parser.add_argument("--v3", type=Path, required=True, help="Path to z-velocity array (.npy or .npz).")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional extra field to chunk alongside the primitives, for example T=sim_data_1/T.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory that will receive chunked field folders. Use <root>/chunked_npz for NPZ-pipeline compatibility.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",
        default=[32],
        help="Chunk size as one integer or three integers, for example --chunk-size 32 or --chunk-size 32 48 24.",
    )
    return parser


def normalize_chunk_size(values: list[int]) -> int | tuple[int, int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return (values[0], values[1], values[2])
    raise ValueError("chunk-size must be one integer or three integers.")


def load_array(path: Path) -> np.ndarray:
    resolved = path.resolve()
    if resolved.suffix == ".npy":
        return np.asarray(np.load(resolved))
    if resolved.suffix == ".npz":
        with np.load(resolved) as data:
            if "arr_0" in data.files:
                return np.asarray(data["arr_0"])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise ValueError(f"{resolved} has multiple arrays; expected arr_0 or a single array entry.")
    raise ValueError(f"Unsupported array file type for {resolved}; expected .npy or .npz.")


def parse_extra_fields(entries: list[str]) -> dict[str, Path]:
    extras: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --extra value {entry!r}; expected NAME=PATH.")
        name, raw_path = entry.split("=", 1)
        field_name = name.strip()
        if not field_name:
            raise ValueError(f"Invalid --extra value {entry!r}; field name cannot be empty.")
        extras[field_name] = Path(raw_path).expanduser()
    return extras


def validate_field_arrays(field_arrays: FieldMap) -> None:
    shapes = {name: array.shape for name, array in field_arrays.items()}
    ndim = {name: array.ndim for name, array in field_arrays.items()}

    bad_dims = {name: rank for name, rank in ndim.items() if rank != 3}
    if bad_dims:
        raise ValueError(f"All arrays must be 3D. Got ranks: {bad_dims}")

    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"All arrays must share the same shape. Got shapes: {shapes}")

    non_finite = [
        name
        for name, array in field_arrays.items()
        if not np.all(np.isfinite(array))
    ]
    if non_finite:
        raise ValueError(f"All arrays must be finite before chunking. Non-finite values found in: {non_finite}")


def save_chunked_primitive_fields(
    *,
    output_dir: Path,
    chunk_size: int | tuple[int, int, int],
    rho: np.ndarray,
    prs: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    extras: FieldMap | None = None,
) -> None:
    field_arrays: FieldMap = {
        "rho": np.asarray(rho),
        "prs": np.asarray(prs),
        "v1": np.asarray(v1),
        "v2": np.asarray(v2),
        "v3": np.asarray(v3),
    }
    if extras:
        for field_name, values in extras.items():
            if field_name in field_arrays:
                raise ValueError(f"Duplicate field name {field_name!r} in extras.")
            field_arrays[field_name] = np.asarray(values)

    validate_field_arrays(field_arrays)
    output_dir.mkdir(parents=True, exist_ok=True)

    for field_name, values in field_arrays.items():
        save_chunked_field(field_name, values, output_dir, target_chunk_size=chunk_size)


def main() -> None:
    args = build_parser().parse_args()
    extra_paths = parse_extra_fields(args.extra)
    chunk_size = normalize_chunk_size(args.chunk_size)

    field_arrays: FieldMap = {
        "rho": load_array(args.rho),
        "prs": load_array(args.prs),
        "v1": load_array(args.v1),
        "v2": load_array(args.v2),
        "v3": load_array(args.v3),
    }
    extras = {field_name: load_array(path) for field_name, path in extra_paths.items()}

    save_chunked_primitive_fields(
        output_dir=args.output_dir.resolve(),
        chunk_size=chunk_size,
        rho=field_arrays["rho"],
        prs=field_arrays["prs"],
        v1=field_arrays["v1"],
        v2=field_arrays["v2"],
        v3=field_arrays["v3"],
        extras=extras,
    )

    print(f"Wrote chunked fields to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

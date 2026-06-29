"""Plot slices, projections, and histograms from `.npz` shock-finder results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

from shockit.npz_pipeline import detect_npz_layout, join_chunked_field


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sim_data_2",
        help="Directory containing plain .npz fields or a chunked_npz layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG files should be written. Defaults to <input_dir>/example_plots.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=30,
        help="Index of the x-slice to visualize.",
    )
    return parser


def load_pyplot():
    config_dir = Path(tempfile.gettempdir()) / "shockit-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_input_field(root: Path, field_name: str) -> np.ndarray:
    layout = detect_npz_layout(root)
    if layout.kind == "plain":
        file_map = {
            "rho": "rho.npz",
            "T": "T.npz",
        }
        return np.load(layout.input_path / file_map[field_name])["arr_0"]
    return join_chunked_field(layout.input_path / field_name)


def load_output_field(root: Path, field_name: str) -> np.ndarray:
    chunked_output_root = root / "shock_chunked_npz"
    if (chunked_output_root / field_name).is_dir():
        return join_chunked_field(chunked_output_root / field_name)

    file_map = {
        "full_shock_mask": "mask_full.npz",
        "mach_temperature": "mach_T.npz",
        "mach_pressure": "mach_prs.npz",
    }
    output_path = root / file_map[field_name]
    if not output_path.exists():
        raise FileNotFoundError(
            f"Could not find {output_path}. Run examples/run_from_npz.py first to generate shock-finder outputs."
        )
    return np.load(output_path)["arr_0"]


def log10_clip(values: np.ndarray, floor: float | None = None) -> np.ndarray:
    positive_floor = floor if floor is not None else np.finfo(float).tiny
    return np.log10(np.clip(values, positive_floor, None))


def positive_finite(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    return flat[np.isfinite(flat) & (flat > 0.0)]


def log_bins(values: np.ndarray, *, default_min: float, default_max: float, count: int = 100) -> np.ndarray:
    positive = positive_finite(values)
    if positive.size == 0:
        return np.logspace(np.log10(default_min), np.log10(default_max), count)

    lower = max(float(np.min(positive)), default_min)
    upper = max(float(np.max(positive)), lower * 10.0)
    return np.logspace(np.log10(lower), np.log10(upper), count)


def plot_slice(
    plt,
    values: np.ndarray,
    *,
    slice_index: int,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    index = max(0, min(slice_index, values.shape[0] - 1))
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(values[index, :, :], origin="lower", cmap=cmap)
    axis.set_title(f"{title} (slice {index})")
    axis.set_xlabel("y index")
    axis.set_ylabel("z index")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_projection(
    plt,
    values: np.ndarray,
    *,
    axis_index: int,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str = "viridis",
    log_scale: bool = False,
) -> None:
    projection_axis = axis_index % values.ndim
    finite_values = np.where(np.isfinite(values), values, 0.0)
    projected = np.sum(finite_values, axis=projection_axis)
    if log_scale:
        projected = log10_clip(projected)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(projected, origin="lower", cmap=cmap)
    axis.set_title(f"{title} (sum axis={projection_axis})")
    axis.set_xlabel("index 1")
    axis.set_ylabel("index 2")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_histograms(
    plt,
    mach_pressure: np.ndarray,
    mach_temperature: np.ndarray,
    *,
    output_dir: Path,
) -> None:
    pressure_values = positive_finite(mach_pressure)
    temperature_values = positive_finite(mach_temperature)
    mach_bins = log_bins(
        np.concatenate((pressure_values, temperature_values)),
        default_min=1.0,
        default_max=100.0,
    )

    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    axis.hist(pressure_values, bins=mach_bins, histtype="step", linewidth=2.5, label="From pressure jump")
    axis.hist(temperature_values, bins=mach_bins, histtype="step", linewidth=2.5, label="From temperature jump")

    guide_x = np.logspace(np.log10(mach_bins[0]), np.log10(mach_bins[-1]), 200)
    axis.plot(guide_x, 1e7 * guide_x ** (-3.0), color="k", linestyle="--", label=r"$M^{-3}$")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Mach")
    axis.set_ylabel("Cell count")
    axis.grid()
    axis.legend()
    figure.savefig(output_dir / "mach_histogram.png", dpi=300)
    plt.close(figure)

    diff_values = positive_finite(np.abs(mach_pressure - mach_temperature))
    diff_bins = log_bins(diff_values, default_min=1e-3, default_max=10.0)

    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    axis.hist(diff_values, bins=diff_bins, histtype="step", linewidth=2.5, color="C2", label="Mach diff")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Absolute Mach difference")
    axis.set_ylabel("Cell count")
    axis.grid()
    axis.legend()
    figure.savefig(output_dir / "mach_diff_histogram.png", dpi=300)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    plt = load_pyplot()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else input_dir / "example_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rho = log10_clip(load_input_field(input_dir, "rho"))
    temperature = log10_clip(load_input_field(input_dir, "T"))
    full_shock_mask = load_output_field(input_dir, "full_shock_mask").astype(float)
    mach_temperature = load_output_field(input_dir, "mach_temperature")
    mach_pressure = load_output_field(input_dir, "mach_pressure")
    mach_diff = np.abs(mach_pressure - mach_temperature)

    plot_slice(
        plt,
        rho,
        slice_index=args.slice_index,
        title="Density",
        colorbar_label="log10(rho)",
        output_path=output_dir / "density_slice.png",
    )
    plot_slice(
        plt,
        temperature,
        slice_index=args.slice_index,
        title="Temperature",
        colorbar_label="log10(T)",
        output_path=output_dir / "temperature_slice.png",
    )
    plot_slice(
        plt,
        full_shock_mask,
        slice_index=args.slice_index,
        title="Full shock mask",
        colorbar_label="Mask",
        output_path=output_dir / "full_shock_mask_slice.png",
    )
    plot_slice(
        plt,
        mach_temperature,
        slice_index=args.slice_index,
        title="Mach from temperature jump",
        colorbar_label="Mach",
        output_path=output_dir / "mach_temperature_slice.png",
    )
    plot_slice(
        plt,
        mach_pressure,
        slice_index=args.slice_index,
        title="Mach from pressure jump",
        colorbar_label="Mach",
        output_path=output_dir / "mach_pressure_slice.png",
    )
    plot_slice(
        plt,
        log10_clip(mach_diff),
        slice_index=args.slice_index,
        title="log10 absolute Mach difference",
        colorbar_label="log10(|Mach_P - Mach_T|)",
        output_path=output_dir / "mach_difference_log10_slice.png",
    )
    plot_projection(
        plt,
        full_shock_mask,
        axis_index=0,
        title="Full shock mask projection",
        colorbar_label="log10 summed mask",
        output_path=output_dir / "full_shock_mask_projection.png",
        log_scale=True,
    )
    plot_projection(
        plt,
        mach_diff,
        axis_index=0,
        title="Mach difference projection",
        colorbar_label="log10 summed |Mach_P - Mach_T|",
        output_path=output_dir / "mach_difference_projection.png",
        log_scale=True,
    )
    plot_histograms(
        plt,
        mach_pressure,
        mach_temperature,
        output_dir=output_dir,
    )

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()

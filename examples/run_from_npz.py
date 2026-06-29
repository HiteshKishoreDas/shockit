"""Run the shock finder on plain or chunked `.npz` simulation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shockit import ShockFinderConfig
from shockit.npz_pipeline import detect_npz_layout, run_npz_shock_finder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sim_data_2",
        help="Directory containing plain .npz fields or a chunked_npz layout.",
    )
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0, help="Adiabatic index.")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing along x.")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing along y.")
    parser.add_argument("--dz", type=float, default=1.0, help="Grid spacing along z.")
    parser.add_argument("--min-mach", type=float, default=1.1, help="Minimum accepted Mach number.")
    parser.add_argument(
        "--sampling-method",
        choices=("nearest_axis", "trilinear"),
        default="trilinear",
        help="Sampling rule used to gather upstream/downstream states.",
    )
    parser.add_argument(
        "--center-score",
        choices=("compression", "mach_pressure", "mach_temperature", "combined"),
        default="compression",
        help="Metric used when center reduction is enabled.",
    )
    parser.add_argument(
        "--shock-width-cells",
        type=int,
        default=1,
        help="Sampling distance from the candidate cell along the shock normal.",
    )
    parser.add_argument(
        "--reduce-to-centers",
        action="store_true",
        help="Reduce connected shock regions to representative center cells.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output during the run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    layout = detect_npz_layout(input_dir)

    config = ShockFinderConfig(
        min_mach=args.min_mach,
        sampling_method=args.sampling_method,
        center_score=args.center_score,
        shock_width_cells=args.shock_width_cells,
        reduce_to_centers=args.reduce_to_centers,
    )
    summary = run_npz_shock_finder(
        input_dir,
        config,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        gamma=args.gamma,
        progress=not args.no_progress,
    )

    print(f"Detected layout: {layout.kind} ({layout.input_path})")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

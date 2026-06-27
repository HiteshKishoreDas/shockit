"""Command-line entry point for shockit."""

from __future__ import annotations

import argparse

from .config import ShockFinderConfig
from .finder import ShockFinder
from .io_hdf5 import load_fluid_cube_from_hdf5
from .output import save_result_hdf5


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Find shocks in a uniform-grid HDF5 snapshot.")
    parser.add_argument("input", help="Input HDF5 file.")
    parser.add_argument("--output", required=True, help="Output HDF5 file.")
    parser.add_argument("--rho-field", default="rho")
    parser.add_argument("--pressure-field", default="pressure")
    parser.add_argument("--vx-field", default="vx")
    parser.add_argument("--vy-field", default="vy")
    parser.add_argument("--vz-field", default="vz")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--dz", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--min-mach", type=float, default=1.1)
    parser.add_argument("--shock-width-cells", type=int, default=1)
    parser.add_argument("--normal-field", choices=("temperature", "pressure"), default="temperature")
    parser.add_argument("--center-score", choices=("compression", "mach_pressure", "mach_temperature", "combined"), default="compression")
    parser.add_argument("--sampling-method", choices=("nearest_axis", "trilinear"), default="nearest_axis")
    parser.add_argument("--chunk-size", type=int, default=128, help="Chunk edge length in cells; use 0 to disable chunking.")
    parser.add_argument("--upstream-pressure-floor", type=float, default=None)
    parser.add_argument("--upstream-temperature-floor", type=float, default=None)
    parser.add_argument("--upstream-density-floor", type=float, default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
    parser.add_argument("--no-reduce-to-centers", action="store_true")
    parser.add_argument("--no-require-gradT-gradRho", action="store_true")
    parser.add_argument("--no-require-density-jump", action="store_true")
    return parser


def main() -> None:
    """Run the shockit CLI."""

    args = build_parser().parse_args()
    cube = load_fluid_cube_from_hdf5(
        filename=args.input,
        rho_field=args.rho_field,
        pressure_field=args.pressure_field,
        vx_field=args.vx_field,
        vy_field=args.vy_field,
        vz_field=args.vz_field,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        gamma=args.gamma,
    )
    config = ShockFinderConfig(
        min_mach=args.min_mach,
        shock_width_cells=args.shock_width_cells,
        normal_field=args.normal_field,
        center_score=args.center_score,
        sampling_method=args.sampling_method,
        chunk_size=None if args.chunk_size == 0 else args.chunk_size,
        upstream_pressure_floor=args.upstream_pressure_floor,
        upstream_temperature_floor=args.upstream_temperature_floor,
        upstream_density_floor=args.upstream_density_floor,
        reduce_to_centers=not args.no_reduce_to_centers,
        require_gradT_gradRho_alignment=not args.no_require_gradT_gradRho,
        require_density_jump=not args.no_require_density_jump,
    )
    progress = False if args.quiet else True
    result = ShockFinder(config).find(cube, progress=progress)
    if not args.quiet:
        total_steps = 10 if config.reduce_to_centers else 9
        print(f"[{total_steps}/{total_steps}] Writing output HDF5")
    save_result_hdf5(args.output, result)


if __name__ == "__main__":
    main()

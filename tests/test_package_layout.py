import importlib

import pytest

from shockit import FluidCube, ShockFinder, ShockFinderConfig


def _tiny_cube() -> FluidCube:
    rho = [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]
    pressure = [[[1.0, 1.0], [1.0, 1.0]], [[4.0, 4.0], [4.0, 4.0]]]
    vx = [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]
    vy = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    vz = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    return FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz)


def test_package_import_smoke() -> None:
    module = importlib.import_module("shockit")
    assert hasattr(module, "FluidCube")
    assert FluidCube is module.FluidCube
    assert ShockFinder is module.ShockFinder
    assert ShockFinderConfig is module.ShockFinderConfig
    assert not hasattr(module, "run_chunked_npz_shock_finder")
    assert not hasattr(module, "run_npz_shock_finder")


def test_chunking_package_exports_explicit_workflow_surface() -> None:
    module = importlib.import_module("shockit.chunking")
    assert hasattr(module, "NpzChunkedInput")
    assert hasattr(module, "NpzChunkedOutput")
    assert hasattr(module, "run_chunked_shock_finder")
    assert hasattr(module, "save_chunked_field")
    assert hasattr(module, "join_chunked_field")
    assert hasattr(module, "join_all_fields")
    assert not hasattr(module, "required_halo")
    assert not hasattr(module, "iter_balanced_slices")


def test_cli_parser_smoke() -> None:
    from shockit.cli import build_parser

    args = build_parser().parse_args(
        [
            "snapshot.h5",
            "--output",
            "shocks.h5",
            "--quiet",
            "--center-score",
            "combined",
            "--sampling-method",
            "nearest_axis",
            "--upstream-pressure-floor",
            "1e-6",
        ]
    )
    assert args.input == "snapshot.h5"
    assert args.output == "shocks.h5"
    assert args.quiet is True
    assert not hasattr(args, "chunk_size")
    assert args.center_score == "combined"
    assert args.sampling_method == "nearest_axis"
    assert args.upstream_pressure_floor == pytest.approx(1e-6)


def test_cli_parser_rejects_removed_chunk_option() -> None:
    from shockit.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["snapshot.h5", "--output", "shocks.h5", "--chunk-size", "64"])


@pytest.mark.parametrize("center_score", ["compression", "mach_pressure", "mach_temperature", "combined"])
def test_center_score_options_do_not_crash(center_score: str) -> None:
    cube = _tiny_cube()
    result = ShockFinder(ShockFinderConfig(center_score=center_score)).find(cube)
    assert result.summary["center_score"] == center_score
    assert "full_shock_cells" in result.summary

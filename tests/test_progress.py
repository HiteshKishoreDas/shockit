from __future__ import annotations

import io
from contextlib import redirect_stdout

from shockit import ShockFinder, ShockFinderConfig
from tests.helpers import make_planar_shock_cube


def test_find_progress_prints_by_default() -> None:
    cube = make_planar_shock_cube()
    output = io.StringIO()

    with redirect_stdout(output):
        ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(cube)

    assert output.getvalue().splitlines() == [
        "[1/8] Computing derived fields",
        "[2/8] Computing gradients",
        "[3/8] Building shock-zone mask",
        "[4/8] Computing shock normals",
        "[5/8] Sampling upstream and downstream jumps",
        "[6/8] Computing Mach fields",
        "[7/8] Applying final shock filters",
        "[8/8] Building summary",
    ]


def test_find_progress_reports_all_steps_with_center_reduction() -> None:
    messages: list[str] = []
    cube = make_planar_shock_cube()

    ShockFinder(ShockFinderConfig(reduce_to_centers=True)).find(
        cube,
        progress=messages.append,
    )

    assert messages == [
        "[1/9] Computing derived fields",
        "[2/9] Computing gradients",
        "[3/9] Building shock-zone mask",
        "[4/9] Computing shock normals",
        "[5/9] Sampling upstream and downstream jumps",
        "[6/9] Computing Mach fields",
        "[7/9] Applying final shock filters",
        "[8/9] Reducing to center cells",
        "[9/9] Building summary",
    ]


def test_find_progress_skips_center_reduction_step_when_disabled() -> None:
    messages: list[str] = []
    cube = make_planar_shock_cube()

    ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(
        cube,
        progress=messages.append,
    )

    assert messages == [
        "[1/8] Computing derived fields",
        "[2/8] Computing gradients",
        "[3/8] Building shock-zone mask",
        "[4/8] Computing shock normals",
        "[5/8] Sampling upstream and downstream jumps",
        "[6/8] Computing Mach fields",
        "[7/8] Applying final shock filters",
        "[8/8] Building summary",
    ]


def test_find_progress_can_be_disabled() -> None:
    cube = make_planar_shock_cube()
    output = io.StringIO()

    with redirect_stdout(output):
        ShockFinder(ShockFinderConfig(reduce_to_centers=False)).find(cube, progress=False)

    assert output.getvalue() == ""

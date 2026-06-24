from __future__ import annotations

from shockit import ShockFinder, ShockFinderConfig
from tests.helpers import make_planar_shock_cube


def test_center_reduced_summary_tracks_component_count_for_planar_shock() -> None:
    cube = make_planar_shock_cube()
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=True, min_mach=1.1)).find(cube)
    assert result.summary["mask_semantics"] == "shock_mask may be center-reduced; full_shock_mask is unreduced"
    assert result.summary["n_connected_components"] >= 1
    assert result.summary["shock_cells"] == result.summary["n_connected_components"]

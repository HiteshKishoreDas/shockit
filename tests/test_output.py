from __future__ import annotations

import h5py

from shockit import ShockFinder, ShockFinderConfig, save_result_hdf5
from tests.helpers import make_planar_shock_cube


def test_output_hdf5_masks_include_descriptions(tmp_path) -> None:
    cube = make_planar_shock_cube()
    result = ShockFinder(ShockFinderConfig(reduce_to_centers=True)).find(cube)
    output_path = tmp_path / "shocks.h5"
    save_result_hdf5(str(output_path), result)

    with h5py.File(output_path, "r") as handle:
        assert "description" in handle["shock_zone_mask"].attrs
        assert "description" in handle["full_shock_mask"].attrs
        assert "description" in handle["shock_mask"].attrs
        assert handle["shock_mask"].attrs["description"].startswith("Final public mask")

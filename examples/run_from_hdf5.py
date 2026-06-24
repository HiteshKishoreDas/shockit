"""Minimal HDF5-based example."""

from shockit import ShockFinder, ShockFinderConfig, load_fluid_cube_from_hdf5, save_result_hdf5


def main() -> None:
    cube = load_fluid_cube_from_hdf5("snapshot.h5")
    result = ShockFinder(ShockFinderConfig(min_mach=1.2)).find(cube)
    save_result_hdf5("shocks.h5", result)


if __name__ == "__main__":
    main()

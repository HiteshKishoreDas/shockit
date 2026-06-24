"""Generate a small synthetic planar-shock HDF5 snapshot."""

import h5py
import numpy as np


def main() -> None:
    n = 32
    rho = np.ones((n, n, n))
    pressure = np.ones((n, n, n))
    vx = np.ones((n, n, n))
    vy = np.zeros((n, n, n))
    vz = np.zeros((n, n, n))

    rho[n // 2 :, :, :] = 2.2857142857142856
    pressure[n // 2 :, :, :] = 4.75
    vx[n // 2 :, :, :] = 0.4375

    with h5py.File("synthetic_planar_shock.h5", "w") as handle:
        handle["rho"] = rho
        handle["pressure"] = pressure
        handle["vx"] = vx
        handle["vy"] = vy
        handle["vz"] = vz


if __name__ == "__main__":
    main()

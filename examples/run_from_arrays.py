"""Minimal array-based example."""

import numpy as np

from shockit import FluidCube, ShockFinder, ShockFinderConfig


def main() -> None:
    n = 16
    rho = np.ones((n, n, n))
    pressure = np.ones((n, n, n))
    vx = np.zeros((n, n, n))
    vy = np.zeros((n, n, n))
    vz = np.zeros((n, n, n))

    pressure[n // 2 :, :, :] = 4.75
    rho[n // 2 :, :, :] = 2.2857142857142856
    vx[: n // 2, :, :] = 1.0
    vx[n // 2 :, :, :] = 0.4375

    cube = FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz)
    result = ShockFinder(ShockFinderConfig(min_mach=1.1)).find(cube)
    print(result.summary)


if __name__ == "__main__":
    main()

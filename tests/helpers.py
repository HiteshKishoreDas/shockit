from __future__ import annotations

import numpy as np

from shockit import FluidCube
from shockit.mach import density_jump_from_mach, pressure_jump_from_mach


def make_planar_shock_cube(mach: float = 2.0, n: int = 32) -> FluidCube:
    gamma = 5.0 / 3.0
    rho1 = 1.0
    p1 = 1.0
    rjump = density_jump_from_mach(mach, gamma)
    pjump = pressure_jump_from_mach(mach, gamma)
    rho2 = rho1 * rjump
    p2 = p1 * pjump
    vx1 = mach * np.sqrt(gamma * p1 / rho1)
    vx2 = vx1 / rjump

    rho = np.full((n, n, n), rho1)
    pressure = np.full((n, n, n), p1)
    vx = np.full((n, n, n), vx1)
    vy = np.zeros((n, n, n))
    vz = np.zeros((n, n, n))

    mid = n // 2
    rho[mid:, :, :] = rho2
    pressure[mid:, :, :] = p2
    vx[mid:, :, :] = vx2
    return FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz, gamma=gamma)


def make_oblique_shock_cube(mach: float = 2.0, n: int = 24) -> FluidCube:
    gamma = 5.0 / 3.0
    rho1 = 1.0
    p1 = 1.0
    rjump = density_jump_from_mach(mach, gamma)
    pjump = pressure_jump_from_mach(mach, gamma)
    rho2 = rho1 * rjump
    p2 = p1 * pjump

    x = np.arange(n)
    y = np.arange(n)
    z = np.arange(n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    signed_distance = (xx + yy) - n

    rho = np.where(signed_distance >= 0, rho2, rho1).astype(float)
    pressure = np.where(signed_distance >= 0, p2, p1).astype(float)
    vx = np.where(signed_distance >= 0, 0.5, 2.0).astype(float)
    vy = np.where(signed_distance >= 0, 0.5, 2.0).astype(float)
    vz = np.zeros((n, n, n))
    return FluidCube(rho=rho, pressure=pressure, vx=vx, vy=vy, vz=vz, gamma=gamma)

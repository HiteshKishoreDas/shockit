Create a standalone Python package for offline shock finding in CFD simulation snapshots.

Package name:
cube_shockfinder

Core design philosophy:
The shock finder should operate on already-extracted uniform 3D NumPy cubes of primitive fluid quantities. Do not make the core algorithm depend on AthenaK, yt, AMR, or HDF5. Instead, design a clean adapter layer that can read HDF5 files and convert fields into uniform 3D arrays. Later, yt can be used externally to interpolate/project SPH or AMR data onto uniform cubes, which can then be passed into this package.

Primary use case:
Offline post-processing of AthenaK/Athena-style hydrodynamic simulation snapshots, especially supersonic turbulence boxes. Version 1 should support only uniform Cartesian grids, ideal-gas hydrodynamics, fixed gamma, and periodic boundaries.

Do not implement:

* AMR
* MHD
* SPH kernels
* yt frontend
* on-the-fly simulation integration
* distributed/chunked/dask processing

Dependencies:

* Python >= 3.10
* numpy
* scipy
* h5py
* pytest

Package structure:
cube_shockfinder/
**init**.py
core.py
fields.py
gradients.py
criteria.py
mach.py
sampling.py
finder.py
io_hdf5.py
output.py
cli.py
tests/
test_mach.py
test_gradients.py
test_planar_shock.py
test_sod.py
test_contact_rejection.py
test_oblique_shock.py
test_hdf5_adapter.py
examples/
run_from_arrays.py
run_from_hdf5.py
make_synthetic_planar_shock.py
pyproject.toml
README.md

Data model:
Implement a dataclass called FluidCube with:

* rho: np.ndarray, shape (Nx, Ny, Nz)
* pressure: np.ndarray, shape (Nx, Ny, Nz)
* vx: np.ndarray
* vy: np.ndarray
* vz: np.ndarray
* dx: float = 1.0
* dy: float = 1.0
* dz: float = 1.0
* gamma: float = 5.0/3.0
* metadata: dict = empty dict

FluidCube should validate:

* all fields have same shape
* fields are 3D
* rho > 0
* pressure > 0
* dx, dy, dz > 0
* gamma > 1

Core output dataclass:
ShockFinderResult with:

* shock_mask: boolean np.ndarray
* shock_zone_mask: boolean np.ndarray
* mach_temperature: np.ndarray
* mach_pressure: np.ndarray
* compression: np.ndarray
* div_v: np.ndarray
* temperature: np.ndarray
* entropy: np.ndarray
* temperature_jump: np.ndarray
* pressure_jump: np.ndarray
* density_jump: np.ndarray
* normal_x: np.ndarray
* normal_y: np.ndarray
* normal_z: np.ndarray
* summary: dict

Core algorithm:
Implement a Skillman-style shock finder for uniform grid data.

Step 1: Derived thermodynamic fields

* temperature proxy:
  T = pressure / rho
* entropy proxy:
  S = log(pressure / rho**gamma)

Step 2: Gradients and divergence
Use second-order centered finite differences with periodic boundaries.

Compute:

* div_v = ∇ · v
* grad_T
* grad_S
* grad_P
* grad_rho

Periodic derivative implementation:
df/dx = [roll(f, -1, axis=0) - roll(f, +1, axis=0)] / (2 dx)

Step 3: Candidate shock zone criteria
A cell is a preliminary shock-zone candidate if:

1. div_v < div_v_threshold

Default:
div_v_threshold = 0.0

2. |grad_T| > grad_T_min

Default:
grad_T_min = 1e-30

3. grad_T · grad_S > 0

This rejects many contact discontinuities.

4. Optional but default enabled:
   grad_T · grad_rho > 0

This helps reject tangential/contact discontinuities.

Step 4: Shock normal
Default shock normal:
n_hat = grad_T / |grad_T|

Add config option:
normal_field = "temperature" or "pressure"

If normal_field = "pressure":
n_hat = grad_P / |grad_P|

Step 5: Upstream/downstream sampling
For each shock candidate cell, sample pre-shock and post-shock states along the shock normal.

Keep v1 simple:

* Convert n_hat to nearest grid direction by choosing the axis with largest absolute normal component.
* Walk plus/minus along that axis.
* Use configurable integer shock_width_cells.

Default:
shock_width_cells = 1

For a candidate cell i:

* sample state_plus from i + shock_width_cells * sign_direction
* sample state_minus from i - shock_width_cells * sign_direction
  using periodic wrapping.

Determine upstream/downstream by pressure:

* downstream is the side with larger pressure.
* upstream is the side with smaller pressure.

Compute jumps:
Pjump = P_down / P_up
Tjump = T_down / T_up
rhojump = rho_down / rho_up

Step 6: Jump consistency criteria
Require by default:

* Pjump > 1
* Tjump > 1
* rhojump > 1

Make these configurable:
require_pressure_jump = True
require_temperature_jump = True
require_density_jump = True

Step 7: Mach number estimation
Implement:

mach_from_pressure_jump(Pjump, gamma):

P2/P1 = [2 gamma M^2 - (gamma - 1)] / (gamma + 1)

So:

M^2 = [(Pjump * (gamma + 1)) + (gamma - 1)] / (2 gamma)

Return NaN if Pjump <= 1.

mach_from_temperature_jump(Tjump, gamma):

T2/T1 =
[(2 gamma M^2 - gamma + 1) ((gamma - 1) M^2 + 2)]
/
[(gamma + 1)^2 M^2]

Implement robust bisection for M in [1, mach_max].

Default:
mach_max = 1000

Return NaN if Tjump <= 1 or if inversion fails.

Step 8: Final shock mask
A cell is in shock_mask if:

* it satisfies shock-zone criteria
* it satisfies jump consistency criteria
* mach_pressure >= min_mach or mach_temperature >= min_mach

Default:
min_mach = 1.1

shock_zone_mask should contain all cells passing the initial compression/gradient criteria before Mach filtering.

Step 9: Optional connected-component reduction
Numerical shocks are often spread over multiple cells. Implement connected-component labeling using scipy.ndimage.label.

Config option:
reduce_to_centers: bool = True

If reduce_to_centers = True:

* label connected components of shock_mask
* for each component, keep only the cell with minimum div_v, i.e. maximum compression
* return this as shock_mask
* keep the unreduced mask internally or as shock_zone_mask / full_shock_mask if useful

Add to result:

* full_shock_mask before center reduction if possible

If reduce_to_centers = False:

* shock_mask is the full filtered shock region.

Step 10: Summary statistics
Return summary dict with:

* grid_shape
* total_cells
* shock_zone_cells
* shock_cells
* shock_fraction
* mach_pressure_min
* mach_pressure_median
* mach_pressure_max
* mach_temperature_min
* mach_temperature_median
* mach_temperature_max
* min_mach
* gamma
* shock_width_cells
* reduce_to_centers

Configuration dataclass:
ShockFinderConfig with fields:

* div_v_threshold: float = 0.0
* grad_T_min: float = 1e-30
* require_gradT_gradS_alignment: bool = True
* require_gradT_gradRho_alignment: bool = True
* normal_field: str = "temperature"
* shock_width_cells: int = 1
* require_pressure_jump: bool = True
* require_temperature_jump: bool = True
* require_density_jump: bool = True
* min_mach: float = 1.1
* mach_max: float = 1000.0
* reduce_to_centers: bool = True

Public API:
The user should be able to do:

```python
import numpy as np
from cube_shockfinder import FluidCube, ShockFinder, ShockFinderConfig

cube = FluidCube(
    rho=rho,
    pressure=pressure,
    vx=vx,
    vy=vy,
    vz=vz,
    dx=dx,
    dy=dy,
    dz=dz,
    gamma=5/3,
)

config = ShockFinderConfig(min_mach=1.2, shock_width_cells=1)
result = ShockFinder(config).find(cube)

shock_mask = result.shock_mask
```

HDF5 adapter:
Implement io_hdf5.py separately from the core finder.

Function:
load_fluid_cube_from_hdf5(
filename,
rho_field="rho",
pressure_field="pressure",
vx_field="vx",
vy_field="vy",
vz_field="vz",
dx=1.0,
dy=1.0,
dz=1.0,
gamma=5/3,
) -> FluidCube

Also support field aliases:

* rho: rho, density, dens
* pressure: pressure, press, prs
* vx: vx, vel1, velocity_x
* vy: vy, vel2, velocity_y
* vz: vz, vel3, velocity_z

But keep it simple. If exact field names are provided, use them.

Output writer:
Implement save_result_hdf5(filename, result), writing:

* shock_mask
* shock_zone_mask
* full_shock_mask if available
* mach_temperature
* mach_pressure
* compression
* div_v
* temperature_jump
* pressure_jump
* density_jump
* normal_x/y/z
* summary as HDF5 attributes where possible

CLI:
Implement command:

cube-shockfind input.h5 --output shocks.h5

Options:
--rho-field
--pressure-field
--vx-field
--vy-field
--vz-field
--dx
--dy
--dz
--gamma
--min-mach
--shock-width-cells
--normal-field temperature/pressure
--no-reduce-to-centers
--no-require-gradT-gradRho
--no-require-density-jump

Example:
cube-shockfind snapshot.h5 
--rho-field rho 
--pressure-field press 
--vx-field vel1 
--vy-field vel2 
--vz-field vel3 
--gamma 1.6666666667 
--min-mach 1.2 
--output shocks.h5

Tests:

Test 1: Mach inversion

* For gamma = 5/3 and known Mach values M = 1.1, 1.5, 2, 5, 10:

  * compute analytic pressure jump
  * invert to Mach
  * assert relative error < 1e-10
  * compute analytic temperature jump
  * invert to Mach
  * assert relative error < 1e-6

Test 2: Periodic gradients

* f = sin(2 pi x / L)
* derivative should be (2 pi / L) cos(2 pi x / L)
* check second-order accuracy approximately.

Test 3: Divergence

* periodic velocity field:
  vx = sin(2 pi x / L)
  vy = sin(2 pi y / L)
  vz = sin(2 pi z / L)
* compare numerical divergence to analytic divergence.

Test 4: Pure contact discontinuity rejection

* Create a pressure-equilibrium density jump:
  pressure constant
  rho jumps
  T changes oppositely
  velocity zero or weak compression
* The algorithm should not identify a strong shock.
* Assert shock_mask has very few or zero cells.

Test 5: Planar shock along x

* Build upstream and downstream states using Rankine-Hugoniot conditions for a known Mach number.
* Create a 3D cube with the discontinuity normal to x.
* Add a converging velocity jump consistent with shock.
* Run finder.
* Assert shock cells are found near the interface.
* Assert median recovered Mach is within about 10–20 percent of input.

Test 6: Oblique planar shock

* Construct a plane with normal approximately (1, 1, 0) / sqrt(2).
* Assign upstream/downstream states by signed distance to plane.
* Run finder.
* Assert shocks are detected near the plane.
* Assert recovered normal direction has positive alignment with true normal on average.

Test 7: Sod-like shock tube

* Either generate approximate left/right states or include a simple fixture.
* The finder should detect the shock and largely avoid marking the contact discontinuity.
* This test can be looser.

Test 8: HDF5 adapter

* Write synthetic rho, pressure, vx, vy, vz arrays to temporary HDF5.
* Load with load_fluid_cube_from_hdf5.
* Assert arrays match.

Implementation priorities:
Milestone 1:

* Implement FluidCube, gradients, mach inversion, and tests.

Milestone 2:

* Implement ShockFinder without connected-component reduction.
* Test planar shock and contact rejection.

Milestone 3:

* Add connected-component center reduction.
* Add summary stats.

Milestone 4:

* Add HDF5 loading and HDF5 result saving.

Milestone 5:

* Add CLI.

Milestone 6:

* Add README and examples.

README should include:

* What the package does.
* Minimal array-based example.
* HDF5 CLI example.
* Explanation of Skillman-style criteria.
* Explanation of Mach estimates from Rankine-Hugoniot jumps.
* Limitations:

  * uniform grid only
  * ideal gas only
  * hydro only
  * periodic boundaries only in v1
  * shock broadening affects Mach estimate
  * false positives possible in turbulent compressions
  * contact rejection is imperfect
  * no AMR/SPH support internally; use external gridding first

Style requirements:

* Use clear type hints.
* Use dataclasses.
* Keep core algorithm independent of HDF5.
* Avoid hidden global state.
* Prefer readable code over premature optimization.
* Add docstrings for all public functions.
* Use pytest.
* Make sure `python -m pytest` passes.
* Make sure the CLI works after `pip install -e .`.

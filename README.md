# Die Bonding Multiphysics Simulation

Finite-element simulation of the epoxy layer during die bonding. A bond head presses the die
down onto a heated substrate; the epoxy between them is squeezed out, heats up, and cures.
The model couples three physics:

1. **Squeeze flow** — incompressible Stokes flow of the epoxy in the bond gap.
2. **Heat transfer** — transient conduction with exothermic heat released by the curing reaction.
3. **Cure kinetics and rheology** — Arrhenius nth-order cure; viscosity rises as the epoxy cures.

Meshing uses [Gmsh](https://gmsh.info); solving uses [DOLFINx](https://github.com/FEniCS/dolfinx)
(FEniCSx) with PETSc.

## Files

| File | Purpose |
|------|---------|
| `01_generate_mesh.py` | Builds meshes with the Gmsh Python API. |
| `02_solve_simulation.py` | Loads the mesh into DOLFINx and runs the coupled time-stepping solver. |
| `03_main.py` | Pipeline entry point: generates the mesh if missing, then runs the solver. Relaunches itself in Docker when DOLFINx is not installed locally. |
| `die_bonding_mesh.msh` | 2D epoxy-gap mesh used by the solver (generated). |
| `die_bonding_results.xdmf` / `.h5` | Time-series results for ParaView (generated). |
| `die_bonding_structure.vtk` | 3D die/epoxy/substrate mesh for visual inspection (generated). |

## How it works

### Meshes (`01_generate_mesh.py`)

- `generate_mesh()` — **the mesh the solver uses.** A 2D cross-section of the epoxy gap
  (default length 2.0 mm, height 50 μm, element size 10 μm), saved as `.msh` with these
  physical tags:

  | Tag | Type | Meaning |
  |-----|------|---------|
  | 1 | surface | Epoxy region |
  | 1 | boundary | Bottom — substrate interface (no-slip, held at stage temperature) |
  | 2 | boundary | Top — die interface (moves down at the bond-head speed) |
  | 3 | boundary | Left/right edges — open outlets (zero traction) |

- `generate_die_bonding_mesh()` — a 3D die + epoxy + substrate model with a refined mesh
  near the bond line. It is used only for visual inspection, not by the solver.
- `visualize_mesh_with_pyvista()` — interactive 3D viewer with a clipping plane.

Running `01_generate_mesh.py` directly builds the 3D model and opens the viewer.

### Solver (`02_solve_simulation.py`)

Units are mm, N, s, MPa, J, K throughout.

At each time step (default Δt = 0.1 s, total 2.0 s):

1. **Temperature** — solve ρC<sub>p</sub> ∂T/∂t − ∇·(k∇T) = ρΔH·dα/dt with implicit (backward) Euler.
   The substrate side is held at 150 °C; the epoxy starts at 25 °C.
2. **Cure degree α** — explicit Euler update of dα/dt = A·exp(−E<sub>a</sub>/RT)·(1−α)<sup>n</sup>,
   clipped to [0, 1].
3. **Squeeze flow** — solve Stokes flow with viscosity μ(α) = μ<sub>0</sub>·exp(C<sub>μ</sub>·α),
   using Taylor–Hood (P2 velocity / P1 pressure) elements and a MUMPS direct solver.

Temperature, cure degree, and pressure are written to `die_bonding_results.xdmf` every step.
The console prints the mean temperature, mean cure degree, and maximum pressure, combined
across all MPI processes.

Material and process parameters (density, heat of reaction, activation energy, bond-head
speed, stage temperature, etc.) are defined at the top of `run_simulation()`.

## Setup

DOLFINx and PETSc are hard to build on macOS, so the solver runs in the official Docker
image. Meshing and 3D viewing can run locally.

**1. Solver (Docker)**

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), start it, then:

```bash
docker pull dolfinx/dolfinx:stable
```

The image includes DOLFINx, PETSc/petsc4py, mpi4py, and Gmsh.

**2. Local Python (optional — for meshing and the 3D viewer)**

```bash
uv venv .venv
uv pip install --python .venv/bin/python gmsh pyvista meshio
```

## Running

Run all commands from the project folder.

**Full pipeline**

```bash
.venv/bin/python 03_main.py
```

If DOLFINx is not installed locally, `03_main.py` automatically reruns itself inside the
`dolfinx/dolfinx:stable` container, so Docker Desktop must be running. You can also call
Docker directly:

```bash
docker run --rm -it -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 03_main.py
```

**Parallel run (MPI)**

```bash
docker run --rm -it -v "$PWD":/work -w /work dolfinx/dolfinx:stable mpirun -n 4 python3 03_main.py
```

**Regenerate the mesh** — `03_main.py` skips meshing if `die_bonding_mesh.msh` exists.
Delete that file after changing mesh parameters.

**View the 3D structure (local)**

```bash
.venv/bin/python 01_generate_mesh.py
```

### Expected output

```
Time: 0.1s | Mean Temp: 147.8°C | Cure: 0.62% | Max Hydro Pressure: 4.8917e-03 MPa
...
Time: 2.0s | Mean Temp: 150.0°C | Cure: 12.23% | Max Hydro Pressure: 7.3415e-03 MPa
[Solver] Simulation complete. Results saved to 'die_bonding_results.xdmf'.
```

### Viewing results

Open `die_bonding_results.xdmf` in [ParaView](https://www.paraview.org/) (keep the `.h5`
file next to it). Choose the *Xdmf3 Reader* if prompted, then select `Temperature`,
`Cure_Degree`, or `Pressure` and play through the time steps.

## Limitations

- **2D model.** The solver uses a 2D cross-section of the gap. The 3D mesh is for
  visualization only.
- **Fixed gap.** The gap height does not shrink as the die moves down; the bond-head speed
  only sets the velocity boundary condition.
- **No convection.** The flow does not carry heat; the temperature equation is
  conduction-only.
- **Duplicate banners under MPI.** `03_main.py` status lines print once per process; solver
  output prints once.

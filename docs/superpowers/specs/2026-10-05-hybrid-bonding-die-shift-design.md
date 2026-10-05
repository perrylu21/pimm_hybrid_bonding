# Hybrid Bonding Die-Shift Study — Design Spec

Date: 2026-10-05
Status: Draft for review

## 1. Goal

Quantify how process, geometry, material, and mechanical parameters affect die shift
(pad overlay error) in die-to-wafer (D2W) Cu–Cu hybrid bonding, for a journal paper.

Method ("PIMM" = FEM + neural-network surrogate):

1. Run a few thousand parametric 3D thermo-mechanical FEM simulations (DOLFINx).
2. Train a physics-informed surrogate (analytic baseline + NN residual) on the FEM data.
3. Use the surrogate for global sensitivity analysis (Sobol indices) and process-window maps.

Success criteria:

- FEM model passes analytic verification (≤1% error) and mesh convergence (≤2% change).
- Main sweep (2,048 runs) completes overnight on one Mac with 4 parallel workers.
- Surrogate achieves holdout R² ≥ 0.95 on the primary outputs (magnification, max misalignment).
- Every paper figure is reproducible from a study YAML + git commit + pinned Docker image.

Constraints:

- Compute: this Mac only (Apple Silicon, Docker Desktop). Target 30–60 s per FEM run.
- FEM stack: `dolfinx/dolfinx` Docker image (DOLFINx 0.11, PETSc). Host `.venv` (Python 3.14)
  for everything that does not need DOLFINx.

Out of scope (possible later extensions): post-bond anneal stage, Cu pad protrusion/stress,
wafer-scale models, HPC execution.

## 2. Architecture

### 2.1 Layout

```
PIMM_hybrid_bonding/
  hbsim/
    materials.py       # fixed constants, unit conversion, bond-layer mixing, analytic baseline
    params.py          # parameter schema (name, unit, range, scale, default); validation; study loading
    sampling.py        # Sobol / LHS / one-at-a-time samplers
    geometry.py        # Gmsh: parametric 3D die + bond layers + wafer patch, extruded hex mesh, tagged
    physics.py         # DOLFINx transient thermal + quasi-static thermo-elastic solves
    postprocess.py     # pad-grid overlay extraction; translation/rotation/magnification fit
    run_one.py         # single run: params -> mesh -> solve -> outputs (executes in Docker)
    runner.py          # sampling, parallel Docker worker pool, resume, dataset collection
    surrogate.py       # analytic baseline + NN residual ensemble; GP baseline (host)
    sensitivity.py     # Sobol indices via SALib on the surrogate (host)
    figures.py         # paper figures from dataset + surrogate (host)
  cli.py               # entry point: run / collect / train / sensitivity / figures
  studies/
    pilot.yaml         # 64 runs, end-to-end smoke test
    main_sweep.yaml    # 2,048-run Sobol training set
    holdout.yaml       # 256-run LHS test set (never used for training)
    lines_*.yaml       # one-at-a-time FEM sweeps
  results/<study>/
    runs/<run_id>/     # params.json, metrics.json, overlay.npz, run.log, optional fields .xdmf
    dataset.parquet    # one row per run: inputs, outputs, status, runtime
    surrogate/         # model.pt, scalers.json, metrics.json, train.log
    figures/
  tests/
  legacy/epoxy_squeeze_flow/   # current 01/02/03 epoxy scripts, reference only
  requirements.txt     # host venv, pinned versions
```

### 2.2 Data flow

```
study YAML ──sampler──▶ parameter sets ──runner (N Docker workers)──▶ run_one.py per set
      ──▶ results/<study>/runs/<run_id>/metrics.json ──collect──▶ dataset.parquet
      ──▶ surrogate.py ──▶ sensitivity.py / figures.py
```

### 2.3 Boundaries

- Only `geometry.py`, `physics.py`, `postprocess.py`, `run_one.py` run inside Docker.
  `params.py`, `materials.py`, `postprocess.py` are shared and import only the standard library
  and numpy (the base image has no PyYAML; `params.load_study` imports it lazily, host only).
  Container-side code reads JSON only.
- `run_one.py` interface: `python -m hbsim.run_one <run_dir>`; reads `<run_dir>/params.json`,
  writes `metrics.json`, `overlay.npz`, `run.log`. Exit code 0 = success.
- `run_id` = first 12 hex chars of SHA-256 of the canonical (sorted-key) JSON of the full
  resolved parameter set. Identical parameters always map to the same directory.
- Each study writes `results/<study>/manifest.json`: resolved YAML, seed, git commit hash
  (plus dirty flag), Docker image digest, start/end time.

### 2.4 Legacy code

`01_generate_mesh.py`, `02_solve_simulation.py`, `03_main.py`, and the current README move to
`legacy/epoxy_squeeze_flow/`. The new package does not import them. A new top-level README
documents the hybrid-bonding workflow.

## 3. Physics Model

### 3.1 Premise

In D2W hybrid bonding the dielectric bond forms within the bond-wave passage. Once a point is
bonded, relative die–wafer displacement there is frozen; later cooling changes stress but not
overlay. Therefore overlay error = die-vs-wafer interface displacement mismatch at the moment
each point bonds, after the bonder's alignment removes rigid translation/rotation.

### 3.2 Geometry and mesh

- Die: Si, `L_die × L_die × t_die`, with a bond layer of thickness `t_bond` on its bottom face.
- Wafer patch: Si, square of side `L_die + 2·margin` (default `margin = 0.5·L_die`) × `t_wafer`,
  with a bond layer of thickness `t_bond` on its top face.
- Bond layers: homogenized dielectric + Cu (volume fraction φ) by rule of mixtures:
  `E_eff = (1−φ)·E_d + φ·E_Cu`, `α_eff = [(1−φ)·E_d·α_d + φ·E_Cu·α_Cu] / E_eff` (Turner),
  `ν_eff = (1−φ)·ν_d + φ·ν_Cu`. Fixed: `E_Cu = 120 GPa`, `ν_Cu = 0.34`, `α_Cu = 16.7 ppm/K`, `ν_d = 0.2`.
- Mesh: structured hexahedra via Gmsh transfinite surfaces + extrusion with layers.
  Die and wafer share the conformal interface plane (needed for the thermal solve).
  Through-thickness layers: ≥ 4 in die, ≥ 4 in wafer, 1 per bond layer. In-plane density set
  by a `mesh_n` parameter (elements along die edge), chosen by the convergence study.
- Physical tags: volumes {1 = die body (bond layer + Si), 2 = wafer body}. Material regions
  (bond layer vs Si) are assigned as DG0 fields by z-coordinate (|z| < t_bond is bond layer), and
  boundary facets are located geometrically — this works unchanged on the die/wafer submeshes.

### 3.3 Step 1 — transient thermal

- Domain: full mesh (die + both bond layers + wafer), perfect thermal contact at the interface.
- Initial condition: die (+ its bond layer) at `T_head`; wafer (+ its bond layer) at `T_chuck`.
- BCs: `die_top` held at `T_head` (bond head); `wafer_bottom` held at `T_chuck` (chuck);
  other faces adiabatic.
- Time window: `[0, t_end]`, `t_end = r_max / v_wave` where `r_max` is the largest distance from
  the wave origin to a die corner. Backward Euler, `n_t = 20` steps (convergence-checked).
- Properties: Si `k = 150 W/(m·K)`, `ρ = 2330 kg/m³`, `c_p = 700 J/(kg·K)`; bond layers by
  volume-weighted mixing with dielectric `k_d = 1.4 W/(m·K)` and Cu `k = 400 W/(m·K)`.

### 3.4 Step 2 — quasi-static thermo-elasticity (die and wafer separately)

Die and wafer are not yet bonded, so they are solved as two bodies: submeshes
(`dolfinx.mesh.create_submesh`) of {die, die_bond} and {wafer, wafer_bond}, with the thermal
solution transferred from the parent mesh at each time step.

Linear isotropic elasticity with eigenstrain `ε* = α·(T − T_ref)·I + ε_bow`, `T_ref = 20 °C`.

- Die bow: through-thickness eigenstrain gradient `ε_bow = −κ·(z − z_mid)·(e_x⊗e_x + e_y⊗e_y)`
  over the whole die body (bond layer + Si), `z_mid` = mid-height of the die body. This field is
  compatible, so a free die is stress-free with curvature κ. `κ = 8·h / L_die²` from the bow
  height `h`; positive h = die bottom convex (center protrudes toward the wafer), which stretches
  the die interface by `κ·(t_die + t_bond)/2`.
- Die BCs: the die is otherwise free. Its 6 rigid modes are removed by point pins on the
  interface plane: (0,0,0) in x, y, z; (L/2,0,0) in y, z; (0,L/2,0) in z. Pinning on the interface
  (not the top) keeps rigid tilt out of the interface in-plane displacement.
- Bond-head compliance: normal spring on `die_top`, traction `−k_head·u_z` (Robin term in the
  stiffness). `k_head → 0` is a free die (full bow strain); `k_head → ∞` is a rigid flat head
  that flattens the bow (bow strain → 0). `k_head` is a swept input (§4.1). Prototype check
  (`h = 5 µm`, `L = 10 mm`): `m_x` = 20.04, 6.90, 0.18 ppm for `k_head` = 0, 1, 10⁴ MPa/mm.
- Bond force: equal and opposite uniform pressure `F / L_die²` on `die_top` (downward) and on the
  die interface face (upward, the wafer reaction) — self-equilibrated; acts via the Poisson effect.
- Wafer BCs: `wafer_bottom` `u_z = 0` and in-plane `u = α_chuck·(T_chuck − T_ref)·(x, y)` (moves
  with the chuck); `wafer_sides` in-plane `u = α_Si·(T_chuck − T_ref)·(x, y)` (continuity with
  the full wafer beyond the patch; side nodes on the bottom edge belong to the bottom BC).
  The wafer also carries the downward pressure `F / L_die²` on its top face within the die
  footprint.
- Solver: stiffness matrices are time-independent → assemble and LU-factor (MUMPS) once per
  body; per time step only the right-hand side is reassembled. P2 (quadratic) hex elements: the
  prototype showed P1 hexes lock in bending at these aspect ratios (bow error ~90%), while P2
  matches the plate solution to 0.1%. Prototype cost: ~195k DOFs, 34–46 s per run.

### 3.5 Step 3 — bond wave and overlay

- Wave origin `(x0, y0)` on the interface (models bond-head tilt); front radius `r(t) = v_wave·t`.
- Pad grid: 21 × 21 points uniformly covering the die interface (inset 2% from edges).
- For pad `i` at `x_i`: lock time `t_i = |x_i − x0| / v_wave`. Overlay vector
  `d_i = u_die,xy(x_i, t_i) − u_wafer,xy(x_i, t_i)`, linearly interpolated between thermal steps.
- Bonder alignment: the bonder aligns die to wafer using marks at process temperature, so it
  removes the rigid translation and rotation present in the *pre-contact* state (t = 0).
  Rigid modes arising during bonding (wave asymmetry, transient heating) are kept — they are
  physical die shift.
- Placement offset `(Δx_p, Δy_p)` and rotation error `Δθ_p` are added analytically in
  post-processing: `d_i += (Δx_p, Δy_p) + Δθ_p·(−y_i, x_i)`. They are not FEM or surrogate inputs.

### 3.6 Verification targets

Verification runs call `physics.simulate(p, lock="t0")`, which locks every pad at contact
(t = 0, both bodies at their uniform initial temperatures) — the `v_wave → ∞` limit — and may use
values outside the sweep ranges (e.g. `F_bond = 0`, `phi_cu = 0`).

- Uniform thermal mismatch: lock at t = 0, `bow_h = 0`, `F_bond = 0`, bond layers set to Si properties
  (`E_d = 130 GPa`, `alpha_d = 2.6 ppm/K`, `phi_cu = 0`), `alpha_chuck = 2.6 ppm/K`. Then each
  body expands uniformly and the corner overlay magnitude equals
  `α_Si·(T_head − T_chuck)·(L_die/2)·√2`. Tolerance 1%.
- Bow only (isothermal at `T_ref`, `k_head = 0`, `F_bond = 0`): fitted magnification equals
  `κ·(t_die + t_bond)/2`. Tolerance 1%.
- Rigid head (`k_head = 10⁴ MPa/mm`), bow only: `|m_x|` < 2% of the free-die value.
- Zero load (all temperatures `T_ref`, `h = 0`, `F = 0`) → overlay = 0 to machine precision.
- Mirror symmetry: `x0 → −x0` flips the sign of `t_x` and leaves `m_x` unchanged.
- Prototype results (2026-10-05): uniform case 3×10⁻⁹ relative error; bow case 0.1%; zero load
  1.5×10⁻¹⁷ mm.

## 4. Parameters, Outputs, Sampling

### 4.1 Swept inputs

| Group | Name | Unit | Range | Scale |
|---|---|---|---|---|
| Thermal | `T_head` | °C | 20–100 | linear |
| Thermal | `T_chuck` | °C | 20–100 | linear |
| Thermal | `v_wave` | mm/s | 5–200 | log |
| Geometry | `L_die` | mm | 2–20 | linear |
| Geometry | `t_die` | µm | 30–775 | log |
| Geometry | `t_wafer` | µm | 300–775 | linear |
| Geometry | `t_bond` | µm | 0.05–1 | log |
| Materials | `E_d` | GPa | 60–200 | linear |
| Materials | `alpha_d` | ppm/K | 0.5–4 | linear |
| Materials | `phi_cu` | – | 0.05–0.4 | linear |
| Materials | `alpha_chuck` | ppm/K | 0.5–5 | linear |
| Mechanical | `bow_h` | µm | −5 to +5 | linear |
| Mechanical | `F_bond` | N | 1–50 | linear |
| Mechanical | `k_head` | MPa/mm | 10⁻³–10⁴ | log |
| Mechanical | `x0`, `y0` | fraction of `L_die` | 0–0.3 | linear |

Fixed: Si isotropic `E = 130 GPa`, `ν = 0.28`, `α = 2.6 ppm/K`; `T_ref = 20 °C`; numerical
settings (`mesh_n`, `n_t`, `margin`) from the convergence study. Any swept parameter can be pinned
in a study YAML with `value:`.

### 4.2 Outputs (`metrics.json`)

- `t_x`, `t_y` [nm] — translation
- `theta` [µrad] — rotation
- `m_x`, `m_y` [ppm] — magnification
- `resid_3sigma` [nm] — 3σ of residual after the linear fit
- `d_max` [nm] — max |d_i| over the pad grid
- `status` (`ok` | `failed`), `error`, `runtime_s`, `n_dofs`

Fit model: `d_i = t + θ·(−y_i, x_i) + diag(m_x, m_y)·x_i + r_i`, least squares over all pads.
`overlay.npz` stores pad coordinates and `d_i`.

### 4.3 Studies

| Study | Sampler | n | Purpose |
|---|---|---|---|
| `pilot` | Sobol | 64 | end-to-end check, timing estimate |
| `main_sweep` | scrambled Sobol (seed 42) | 2,048 | surrogate training |
| `holdout` | LHS (seed 7) | 256 | surrogate test, never trained on |
| `lines_<param>` | one-at-a-time | ~15 per param | FEM trend figures, surrogate check |

YAML schema:

```yaml
name: main_sweep
sampler: {type: sobol, n: 2048, seed: 42}   # sobol | lhs | line
workers: 4
timeout_s: 600
image: dolfinx/dolfinx@sha256:<digest>
params:
  T_head:  {range: [20, 100], unit: degC}
  v_wave:  {range: [5, 200], unit: mm/s, scale: log}
  t_bond:  {value: 0.2, unit: um}            # pinned
```

For `line` studies: `sampler: {type: line, param: T_head, n: 15}`; other parameters at their
defaults.

## 5. Surrogate and Sensitivity

### 5.1 Physics-informed surrogate

For each output `y`: `ŷ = y_analytic(p) + NN(p)`.

- `y_analytic`: closed-form baseline from §3.6. Magnification
  `m_a = α_Si·(T_head − T_ref) − α_chuck·(T_chuck − T_ref) + κ·(t_die + t_bond)/2` (thin bond
  layers neglected; wafer interface assumed to follow the chuck; free-die bow, i.e. the
  `k_head` reduction is left to the NN) gives `m_x = m_y = m_a` and
  `d_max = |m_a|·(L_die/2)·√2`. Baseline is zero for `t_x`, `t_y`, `theta`, `resid_3sigma`.
- NN: MLP (PyTorch), 4 hidden layers × 128, SiLU, Adam, early stopping on a 10% validation
  split of `main_sweep`. Inputs scaled to [0, 1] (log-scale parameters log-transformed first);
  residual targets standardized.
- Ensemble of 5 networks with different seeds; report mean and spread.
- Baseline comparison: Gaussian process (scikit-learn, Matérn 5/2 + white noise) on the same
  inputs and targets.

Evaluation on `holdout`: R² and RMSE per output, parity plots, learning curve
(n = 128, 256, 512, 1024, 2048).

### 5.2 Sensitivity and figures

- Sobol first-order `S_i`, total `S_Ti`, and second-order `S_ij` indices per output (SALib
  Saltelli sampling with second order, N = 2¹⁴ → N·(2D+2) ≈ 557k surrogate evaluations),
  95% bootstrap confidence intervals.
- Interaction ranking: report `S_Ti − S_i` (each input's share of variance through interactions)
  and the largest `S_ij` pairs. Hypotheses to test, not assume: bow × k_head, ΔT × v_wave,
  bow × v_wave. Prototype hint: bond-wave asymmetry effects are small (x0 = 0.2 gave
  t_x ≈ 0.007 nm) because thermal diffusion through the die (~ms) is much faster than the wave
  (~0.1 s), so v_wave interactions may turn out weak — report whichever the data show.
- Figures: stacked bar of `S_i` and `S_Ti − S_i` per input; `S_ij` heatmap per output.
- Process-window contour maps of `d_max` over chosen parameter pairs (default
  `ΔT = T_head − T_chuck` vs `bow_h`), with a configurable spec line (default 100 nm).
- Line-study figures: FEM points with surrogate prediction band overlaid.

## 6. Error Handling

- Each run is a separate `docker run` with a timeout (default 600 s).
- Failures (non-zero exit, exception, timeout, non-converged solve, non-finite outputs) produce
  `metrics.json` with `status: failed` and the error message; the sweep continues.
- Pre-solve validation: values inside declared ranges; geometric sanity (`t_die > 10·t_bond`,
  `|bow_h| < L_die/100`).
- `cli.py run` skips runs with an existing `status: ok`; `--retry-failed` reruns failed ones.
- `dataset.parquet` is always rebuilt from `runs/`; per-run files are the source of truth.
- Surrogate training excludes failed runs and logs how many were excluded.

## 7. Testing

- `tests/test_verification.py` (Docker): §3.6 analytic and symmetry checks.
- `tests/test_convergence.py` (Docker, slow, marked): refine `mesh_n`, `nz_die`, `nz_wafer`,
  and `n_t` until the key outputs change < 2%, at the default case and the thin-layer extreme
  (`t_bond = 0.05 µm`, `t_die = 30 µm`, `L_die = 20 mm`, aspect ratio ~10⁴); writes
  `results/convergence.csv` for the paper's appendix. It also compares the thin-layer case with
  the bond layer replaced by Si properties: the difference bounds the layer's own effect and
  confirms the thin elements do not corrupt the solution.
- `tests/test_postprocess.py` (host): fit recovers synthetic translation, rotation,
  magnification exactly.
- `tests/test_params.py` (host): range/scale mapping round-trips; samplers reproducible by seed;
  `run_id` stable.
- `tests/test_surrogate.py` (host): save/load round-trip; fits a known analytic function.
- `pilot` study as end-to-end integration check before `main_sweep`.

## 8. Environment and Reproducibility

- FEM image pinned by digest in each study YAML
  (`dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8`).
  No custom image: container-side code needs only numpy, gmsh, DOLFINx (all in the base image).
- Host `.venv`: gmsh, pyvista, numpy, pandas, pyarrow, scipy, scikit-learn, SALib, torch,
  matplotlib, pytest; versions pinned in `requirements.txt`.
- Git repository initialized; `results/`, `.venv/`, meshes, and field outputs are git-ignored.
  `manifest.json` records commit hash and dirty flag for every study.

## 9. Risks

- 50 nm bond layers under ~0.5 mm in-plane elements create extreme aspect ratios (~10⁴).
  P2 hexes handled the extreme case in the prototype (46 s, no solver failure). Mitigation if the
  convergence test fails: raise `mesh_n`; last resort, model each bond layer as a membrane
  (surface) layer on its own body. A spring (Robin) interface between die and wafer is not used:
  before bonding the bodies are not connected, and a spring cannot represent the layer's own
  thermal expansion.
- Run time may exceed 60 s at the converged mesh. Mitigation: pilot study measures it; reduce
  `main_sweep` to 1,024 runs or coarsen the wafer patch if needed.
- PyTorch on Python 3.14 / macOS arm64: verified available (torch 2.14.1, 2026-10-05).
  Fall back to a 3.12/3.13 venv only if a later dependency lacks 3.14 wheels.

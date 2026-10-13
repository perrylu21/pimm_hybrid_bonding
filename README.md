# Hybrid Bonding Die-Shift Study

Parametric study of die shift (pad overlay error) in die-to-wafer Cu–Cu hybrid bonding.

1. **FEM** — a 3D thermo-mechanical model of the bonding moment (DOLFINx, in Docker) computes
   pad overlay for one parameter set.
2. **Sweep** — thousands of FEM runs over 16 process, geometry, material, and mechanical
   parameters, sampled with Sobol / Latin-hypercube designs.
3. **Surrogate** — an analytic overlay formula plus a neural-network correction, trained on the
   FEM runs.
4. **Sensitivity** — Sobol indices (first-order, total, pairwise) computed on the surrogate.

Design spec: `docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md`.

## Layout

| Path | Contents |
|------|----------|
| `hbsim/params.py` | Parameter schema (names, units, ranges, defaults), validation, study loading |
| `hbsim/materials.py` | Material constants, bond-layer mixing, analytic overlay baseline |
| `hbsim/sampling.py` | Sobol, Latin-hypercube, and one-at-a-time samplers |
| `hbsim/geometry.py` | Gmsh structured hex mesh of die + bond layers + wafer patch |
| `hbsim/physics.py` | Heat conduction, thermo-elasticity, bond-wave locking (DOLFINx) |
| `hbsim/postprocess.py` | Overlay fit: translation, rotation, magnification, residual |
| `hbsim/run_one.py` | Runs one simulation directory (inside Docker) |
| `hbsim/runner.py` | Parallel Docker runs, resume, `dataset.parquet` collection |
| `hbsim/surrogate.py` | Analytic baseline + MLP ensemble; Gaussian-process comparison |
| `hbsim/sensitivity.py` | Sobol indices with SALib |
| `hbsim/figures.py` | Paper figures (PDF) |
| `cli.py` | Command-line entry point |
| `studies/*.yaml` | Study definitions (`*_fast.yaml`: quick 1,024-run route) |
| `results/<study>/` | Run folders, `dataset.parquet`, `manifest.json`, surrogate, figures (not in git) |
| `legacy/epoxy_squeeze_flow/` | Earlier epoxy squeeze-flow scripts, kept for reference |

## Setup

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   Each run needs about 3.6 GB at the default mesh (`mem_per_run_gb` in the study file), and the
   runner starts only as many parallel runs as fit in Docker's memory limit
   (Settings → Resources): 8 GB → 2 runs, 12 GB → 3, 16 GB → 4.
2. Pull the FEM image (pinned by digest in every study file):

   ```bash
   docker pull dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
   ```

3. Create the host environment:

   ```bash
   uv venv -p 3.14 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

## Quick route: 1,024-run study (recommended first)

1,024 training runs on the coarser mesh (`mesh_n` 12, within 0.2% of the finest mesh) instead of
2,048 on the default mesh. Total time is about 2.5–3 h instead of about 14 h. If the surrogate is
accurate enough (step 5), use these results; otherwise extend the sweep (step 6).

The swept inputs use the temperature difference `dT_head = T_head − T_chuck` and the die
curvature `kappa` (1/m; bow height = kappa·L²/8). Results made before this change (with `T_head`
and `bow_h` inputs) are not compatible: move them aside before step 3:

```bash
mkdir -p results/old_T_head_bow_h
mv results/main_sweep_fast results/holdout_fast results/pilot results/old_T_head_bow_h/ 2>/dev/null
mv results/lines_* results/old_T_head_bow_h/ 2>/dev/null
```

Run every command from the project folder:

```bash
cd /Users/perrylu21/project/claude_code/PIMM_hybrid_bonding
```

**Step 1. Start Docker Desktop** and check it responds:

```bash
open -a Docker
docker info --format 'Docker memory: {{.MemTotal}} bytes'
```

**Step 2. (Optional, ~4 min) Run the tests:**

```bash
.venv/bin/python -m pytest
docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider
```

Both should end with `passed` and no `failed`.

**Step 3. Training set: 1,024 runs (~2–2.5 h).**

```bash
.venv/bin/python cli.py run studies/main_sweep_fast.yaml
```

Progress prints one line per run (`[main_sweep_fast] 37/1024 <run_id> ok`). With 8 GB of Docker
memory it runs 3 in parallel. You can stop with Ctrl-C and rerun the same command later: finished
runs are kept. When it ends, check that nothing failed:

```bash
.venv/bin/python cli.py collect results/main_sweep_fast
```

Expected: `ok    1024`. If some show `failed`, read `results/main_sweep_fast/runs/<run_id>/run.log`,
then rerun them with `.venv/bin/python cli.py run studies/main_sweep_fast.yaml --retry-failed`.

**Step 4. Test set: 128 runs (~15–20 min).** These are never used for training; they measure how
accurate the surrogate is.

```bash
.venv/bin/python cli.py run studies/holdout_fast.yaml
.venv/bin/python cli.py collect results/holdout_fast
```

Expected: `ok    128`.

**Step 5. Train the surrogate (a few minutes).**

```bash
.venv/bin/python cli.py train --train results/main_sweep_fast --test results/holdout_fast --gp --learning-curve
```

It prints the holdout accuracy of each output, for example `m_x  R2=0.9950  RMSE=2.1`. Then look
at the learning curve (holdout error at 128, 256, 512 and 1,024 training runs):

```bash
.venv/bin/python -c "
import pandas as pd
c = pd.read_csv('results/main_sweep_fast/surrogate/learning_curve.csv')
print(c[c.output.isin(['m_x', 'd_max'])].pivot(index='n_train', columns='output', values='rmse').round(3))"
```

The surrogate is good enough when both hold:

- `R2` ≥ 0.95 for `m_x` and `d_max`.
- RMSE at 1,024 is less than 10% lower than at 512 (the curve has flattened, so more runs would not
  help much).

`--gp` also fits a Gaussian-process (GP) model on the same data (~4 min for 1,024 runs) and saves it
to `results/main_sweep_fast/surrogate_gp/`. Its holdout scores are in
`results/main_sweep_fast/surrogate/metrics.json` under `"gp"`. Steps 7 and 8 work with either
model; running both and comparing is a robustness check worth reporting.

**Step 6. Only if step 5 fails: extend to 2,048 runs (~2–2.5 h more).** In
`studies/main_sweep_fast.yaml`, change `n: 1024` to `n: 2048`, then rerun steps 3 and 5. The first
1,024 runs are reused, so only the new 1,024 execute.

**Step 7. Sensitivity analysis (~1 min).**

```bash
.venv/bin/python cli.py sensitivity --surrogate results/main_sweep_fast/surrogate
```

It prints the five most influential parameters for `m_x` and `d_max` (`S1` = effect alone, `ST` =
total effect, `interaction` = `ST − S1`). The full tables are written to
`results/main_sweep_fast/surrogate/sobol_first_total.csv` and `sobol_second_order.csv`.

Same analysis with the GP model (~1.5 min); tables are written to `surrogate_gp/`:

```bash
.venv/bin/python cli.py sensitivity --surrogate results/main_sweep_fast/surrogate_gp
```

**Step 8. Figures.**

```bash
.venv/bin/python cli.py figures --surrogate results/main_sweep_fast/surrogate --test results/holdout_fast
```

PDFs are written to `results/main_sweep_fast/figures/`: parity plots, Sobol bars, pairwise
interaction heatmaps, learning curve, and two process-window maps (`dT_head` × `kappa`,
`kappa` × `k_head`). For the GP model, pass `--surrogate results/main_sweep_fast/surrogate_gp`;
its figures go to `results/main_sweep_fast/figures_gp/` (the learning curve is MLP-only).

**Step 9. (Optional, ~30 min) FEM trend checks.** One-at-a-time FEM sweeps of five parameters,
drawn against the surrogate's prediction:

```bash
for f in studies/lines_*.yaml; do .venv/bin/python cli.py run "$f"; done
.venv/bin/python cli.py figures --surrogate results/main_sweep_fast/surrogate --test results/holdout_fast \
    --lines results/lines_*
```

## Full workflow (2,048 runs)

Run everything from the project folder.

```bash
# 1. Tests: host, then FEM (inside Docker)
.venv/bin/python -m pytest
docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest

# 2. Pilot (64 runs): checks the pipeline and measures run time
.venv/bin/python cli.py run studies/pilot.yaml

# 3. Training and test sets (main_sweep: 2,048 runs of ~40 s: ~11 h with 2 workers, ~6 h with 4)
.venv/bin/python cli.py run studies/main_sweep.yaml
.venv/bin/python cli.py run studies/holdout.yaml
for f in studies/lines_*.yaml; do .venv/bin/python cli.py run "$f"; done

# 4. Surrogate, sensitivity, figures
.venv/bin/python cli.py train --train results/main_sweep --test results/holdout --gp --learning-curve
.venv/bin/python cli.py sensitivity --surrogate results/main_sweep/surrogate
.venv/bin/python cli.py figures --surrogate results/main_sweep/surrogate --test results/holdout \
    --lines results/lines_*
```

**Interrupting and resuming.** Each run is stored in `results/<study>/runs/<run_id>/`, named by
a hash of its parameters. Re-running a study skips finished runs; add `--retry-failed` to rerun
failures. `dataset.parquet` is always rebuilt from the run folders
(`cli.py collect results/<study>`).

**Reproducibility.** Each study writes `manifest.json` with the study file, seed, git commit
(and whether the working tree had uncommitted changes), and Docker image digest. Commit before
running a study you intend to report.

## Study files

```yaml
name: main_sweep
sampler: {type: sobol, n: 2048, seed: 42}   # sobol (n = power of 2) | lhs | line
workers: 4
threads_per_worker: 2
mem_per_run_gb: 3.6          # peak memory per run; parallel runs are capped to fit Docker's memory
timeout_s: 600
image: dolfinx/dolfinx@sha256:...
params:
  dT_head: {range: [10, 40]}   # narrow a sweep range (must lie inside the schema range)
  t_bond: {value: 0.2}         # pin a parameter
numerics:
  mesh_n: 28                   # override a numerical setting
```

Parameters not listed under `params` are swept over their full schema range. For a line study:
`sampler: {type: line, param: dT_head, n: 15}` — other parameters stay at their defaults.

## Parameter glossary

Defined in code at `hbsim/params.py` (`SWEPT`, `NUMERICS`), `hbsim/materials.py` (constants) and
`hbsim/postprocess.py` (`OUTPUTS`); the physics is in the design spec, §3.

### Swept inputs (16)

| Name | Meaning | Unit | Range | Default | Where it acts in the model |
|------|---------|------|-------|---------|----------------------------|
| `dT_head` | Bond-head temperature minus chuck temperature (`T_head = T_chuck + dT_head`) | °C | 0–60 | 20 | Die starts at `T_head`; die top held at `T_head` during bonding |
| `T_chuck` | Chuck (wafer) temperature | °C | 20–60 | 40 | Wafer starts at `T_chuck`; wafer bottom held at `T_chuck`; sets wafer and chuck thermal expansion |
| `v_wave` | Bond-wave speed: how fast the bonded area spreads from its start point (log scale) | mm/s | 5–200 | 30 | A pad locks when the wave reaches it: lock time = distance from start / `v_wave` |
| `L_die` | Die edge length (square die) | mm | 2–20 | 10 | Die size; wafer patch is `2 × L_die` wide |
| `t_die` | Die silicon thickness (log scale) | µm | 30–775 | 100 | Die stiffness; with `kappa`, sets bending strain `kappa·t_die/2` |
| `t_wafer` | Wafer silicon thickness | µm | 300–775 | 775 | Wafer stiffness and heat path to the chuck |
| `t_bond` | Thickness of each bond layer (dielectric + Cu pads), on die and on wafer (log scale) | µm | 0.05–1 | 0.2 | Thin layer on each bonding surface |
| `E_d` | Young's modulus of the bond dielectric (SiO₂ ≈ 70, SiCN ≈ 150–200) | GPa | 60–200 | 100 | Bond-layer stiffness (mixed with Cu by `phi_cu`) |
| `alpha_d` | Thermal expansion of the bond dielectric | ppm/K | 0.5–4 | 1.5 | Bond-layer expansion (mixed with Cu by `phi_cu`) |
| `phi_cu` | Cu pad area fraction in the bond layer | – | 0.05–0.4 | 0.2 | Mixing weight between dielectric and Cu properties |
| `alpha_chuck` | Thermal expansion of the chuck (Si = 2.6) | ppm/K | 0.5–5 | 2.6 | Wafer bottom moves with the chuck: `alpha_chuck·(T_chuck − 20 °C)` |
| `kappa` | Die curvature before bonding; > 0 means the die bottom bulges toward the wafer. Bow height = `kappa·L_die²/8` | 1/m | −0.5 to +0.5 | 0 | Stretches (> 0) or compresses (< 0) the die bottom surface by `kappa·t_die/2` |
| `F_bond` | Bond force applied by the head | N | 1–50 | 10 | Pressure `F_bond / L_die²` on die top and bottom (Poisson effect) |
| `k_head` | Bond-head stiffness against die bending: ≈ 0 free die, large = rigid flat head (log scale) | MPa/mm | 10⁻³–10⁴ | 1 | Spring on the die top; a stiff head flattens a curved die |
| `x0`, `y0` | Bond-wave start point, offset from die center as a fraction of `L_die` (models head tilt) | – | 0–0.3 | 0 | Origin of the bond wave |

### Fixed constants (not swept)

| Constant | Value | Notes |
|----------|-------|-------|
| Reference temperature `T_REF` | 20 °C | Pads are patterned and aligned at this temperature; expansion is measured from it |
| Silicon (die and wafer) | E 130 GPa, ν 0.28, α 2.6 ppm/K, k 150 W/(m·K), ρc_p 1.63 MJ/(m³·K) | Isotropic |
| Copper (bond pads) | E 120 GPa, ν 0.34, α 16.7 ppm/K, k 400 W/(m·K), ρc_p 3.45 MJ/(m³·K) | |
| Bond dielectric | ν 0.2, k 1.4 W/(m·K), ρc_p 1.6 MJ/(m³·K) | E and α are swept (`E_d`, `alpha_d`) |
| Bond-layer mixing | E, ν, k, ρc_p by volume fraction; α by stiffness-weighted average (Turner) | Weight = `phi_cu` |

### Numerical settings (`numerics:` in a study file)

| Name | Meaning | Default |
|------|---------|---------|
| `mesh_n` | Elements along the die edge (even). `*_fast.yaml` studies use 12 | 20 |
| `n_ring` | Elements across the wafer margin around the die | 6 |
| `nz_die`, `nz_wafer` | Elements through the die / wafer silicon thickness | 4, 4 |
| `n_t` | Thermal time steps during the bond wave (geometric, starting at 1 µs) | 20 |
| `margin_frac` | Wafer margin on each side of the die, as a fraction of `L_die` | 0.5 |
| `deg` | Displacement element order (2 = quadratic; 1 locks in bending) | 2 |
| `n_pads` | Pad grid where overlay is measured: `n_pads × n_pads` points covering the die | 21 |

### Outputs per run

| Output | Unit | Meaning |
|--------|------|---------|
| `t_x`, `t_y` | nm | Translation of die pads relative to wafer pads. Set mostly by the model's die-center anchor; not reliable as die-shift results |
| `theta` | µrad | Rotation (same caveat as translation) |
| `m_x`, `m_y` | ppm | Magnification: scale mismatch between die and wafer pad patterns. 1 ppm = 1 nm per mm from the die center |
| `resid_3sigma` | nm | 3σ of the misalignment left after removing translation, rotation and magnification (non-uniform distortion) |
| `d_max` | nm | Largest pad misalignment anywhere on the die |

The bonder's alignment removes translation and rotation present before contact; everything
measured after that comes from bonding. Bonder placement offset and rotation error are not
simulated; add them afterwards with `hbsim.postprocess.apply_placement`.

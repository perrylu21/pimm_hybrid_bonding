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

The Gaussian-process comparison scores are in `results/main_sweep_fast/surrogate/metrics.json`
under `"gp"`.

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

**Step 8. Figures.**

```bash
.venv/bin/python cli.py figures --surrogate results/main_sweep_fast/surrogate --test results/holdout_fast
```

PDFs are written to `results/main_sweep_fast/figures/`: parity plots, Sobol bars, pairwise
interaction heatmaps, learning curve, and two process-window maps (`dT_head` × `kappa`,
`kappa` × `k_head`).

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

## Outputs per run

| Output | Unit | Meaning |
|--------|------|---------|
| `t_x`, `t_y` | nm | Translation of die pads relative to wafer pads |
| `theta` | µrad | Rotation |
| `m_x`, `m_y` | ppm | Magnification (scale mismatch) |
| `resid_3sigma` | nm | 3σ of the overlay left after removing the linear fit |
| `d_max` | nm | Largest pad misalignment |

Bonder placement offset and rotation error are not simulated; add them afterwards with
`hbsim.postprocess.apply_placement`.

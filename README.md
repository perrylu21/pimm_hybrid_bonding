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
| `studies/*.yaml` | Study definitions |
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

## Workflow

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
  T_head: {range: [30, 80]}    # narrow a sweep range (must lie inside the schema range)
  t_bond: {value: 0.2}         # pin a parameter
numerics:
  mesh_n: 28                   # override a numerical setting
```

Parameters not listed under `params` are swept over their full schema range. For a line study:
`sampler: {type: line, param: T_head, n: 15}` — other parameters stay at their defaults.

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

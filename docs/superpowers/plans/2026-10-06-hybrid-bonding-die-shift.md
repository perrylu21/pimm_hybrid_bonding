# Hybrid Bonding Die-Shift Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the epoxy squeeze-flow scripts with a reproducible pipeline that runs thousands of 3D thermo-mechanical FEM simulations of D2W hybrid bonding, trains a physics-informed surrogate on them, and produces Sobol sensitivity results and paper figures for die shift.

**Architecture:** A Python package `hbsim/`. Container-side modules (`geometry`, `physics`, `run_one`) need DOLFINx and run in the pinned `dolfinx/dolfinx` Docker image; shared modules (`params`, `materials`, `postprocess`) are pure Python/numpy; host-side modules (`sampling`, `runner`, `surrogate`, `sensitivity`, `figures`, `cli.py`) run in the Mac `.venv`. Each FEM run is one Docker container writing `params.json` → `metrics.json` in its own hash-named folder; the runner parallelizes, resumes, and collects a Parquet dataset.

**Tech Stack:** DOLFINx 0.11 + PETSc/MUMPS + Gmsh 4.15 (Docker), Python 3.14 host with numpy, scipy, pandas/pyarrow, PyYAML, PyTorch 2.14, scikit-learn 1.9, SALib 1.6, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md`

## Global Constraints

- FEM image: `dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8` (DOLFINx 0.11.0, PETSc 3.25.1, Gmsh 4.15.2, Python 3.12, pytest 9.0.3, no PyYAML, no pandas).
- Host: `.venv` on Python 3.14, versions pinned in `requirements.txt` (torch 2.14.1 verified on macOS arm64).
- Code that runs in the container (`params.py`, `materials.py`, `postprocess.py`, `geometry.py`, `physics.py`, `run_one.py`) imports only the standard library, numpy, gmsh, and DOLFINx. It reads JSON, never YAML.
- Model units: mm, MPa (N/mm²), N, s, °C, J. User-facing parameter units are listed in `hbsim/params.py`.
- Displacement elements are P2 hexahedra (P1 locks in bending: ~90% bow error in the prototype).
- Every FEM run is serial (`MPI.COMM_SELF`); parallelism is 4 concurrent containers with `OMP_NUM_THREADS=2`.
- Tests that need DOLFINx call `pytest.importorskip("dolfinx")`; tests that need host-only packages skip themselves in the container. Run both suites:
  - host: `.venv/bin/python -m pytest`
  - container: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider`
- Docker Desktop must be running for container steps (`open -a Docker` on macOS).
- Commit messages end with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

These conditions are implied by the spec but easy to miss. Each has a test in the owning task.

1. **Docker not running or image missing** — `cli.py run` must fail immediately with "start Docker Desktop" / "docker pull …", not record 2,048 identical failures. (Task 8: `test_docker_preflight_messages`)
2. **Study file edited after some runs exist** — runs from the old sample set must not leak into `dataset.parquet`. (Task 8: `test_edited_study_ignores_stale_runs`)
3. **Run interrupted (Ctrl-C, crash, out-of-memory kill)** — folders with `params.json` but no or unreadable `metrics.json` are treated as not run and executed on resume; containers killed without writing metrics are recorded as failed with the exit code. (Task 8: `test_crash_without_metrics_is_recorded_and_retry_works`, `test_corrupt_metrics_file_is_treated_as_not_run`)
4. **Many failed runs** — failures may cluster (e.g. at the thin-layer extreme) and silently bias the surrogate; training must warn above 2% unusable runs. (Task 10: `test_high_failure_rate_warns`)
5. **Surrogate loaded after the parameter schema changed** — loading must refuse rather than silently mis-map inputs. (Task 10: `test_load_rejects_mismatched_schema`)

## File Structure

```
PIMM_hybrid_bonding/
  hbsim/__init__.py        empty package marker
  hbsim/materials.py       constants, unit conversion, bond-layer mixing, analytic baseline   (Task 2)
  hbsim/params.py          schema, scaling, validation, run_id, Study loading                (Task 2)
  hbsim/sampling.py        Sobol / LHS / line samplers                                       (Task 3)
  hbsim/postprocess.py     pad grid, overlay fit, metrics, placement                         (Task 4)
  hbsim/geometry.py        Gmsh structured hex model                                         (Task 5)
  hbsim/physics.py         thermal + elastic solves, bond-wave locking                       (Task 6)
  hbsim/run_one.py         one run directory -> metrics.json                                 (Task 7)
  hbsim/runner.py          parallel Docker runs, resume, collect                             (Task 8)
  hbsim/surrogate.py       baseline + MLP ensemble, GP baseline, evaluation                  (Task 10)
  hbsim/sensitivity.py     Sobol indices via SALib                                           (Task 10)
  hbsim/figures.py         paper figures                                                     (Task 11)
  cli.py                   run / collect (Task 8); train / sensitivity / figures (Task 11)
  studies/*.yaml           pilot, main_sweep, holdout, lines_*                               (Task 3)
  tests/                   one test file per module + convergence                            (each task)
  pytest.ini, requirements.txt, .gitignore                                                    (Task 1)
  legacy/epoxy_squeeze_flow/   old 01/02/03 scripts + old README                              (Task 1)
  README.md                new workflow documentation                                         (Task 12)
```

---

### Task 1: Project restructure and host environment

**Files:**
- Move: `01_generate_mesh.py`, `02_solve_simulation.py`, `03_main.py`, `README.md` → `legacy/epoxy_squeeze_flow/`
- Create: `hbsim/__init__.py`, `tests/conftest.py`, `pytest.ini`, `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: importable empty package `hbsim`; `pytest` configured with `testpaths = tests` and a `slow` marker excluded by default; host `.venv` with all dependencies.

- [ ] **Step 1: Move the legacy scripts**

```bash
mkdir -p legacy/epoxy_squeeze_flow
git mv 01_generate_mesh.py 02_solve_simulation.py 03_main.py README.md legacy/epoxy_squeeze_flow/
```

- [ ] **Step 2: Create the package marker, test config, and requirements**

`hbsim/__init__.py`: empty file.

`tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: long-running FEM tests (mesh convergence); run with -m slow
addopts = -m "not slow"
```

`requirements.txt`:

```text
# Host (.venv, Python 3.14) dependencies. FEM dependencies live in the DOLFINx Docker image.
gmsh==4.15.2
pyvista==0.49.0
numpy==2.5.3
scipy==1.18.1
pandas==3.0.6
pyarrow==25.0.1
pyyaml==6.0.3
scikit-learn==1.9.1
SALib==1.6.0
torch==2.14.1
matplotlib==3.11.2
pytest==9.1.1
```

Append to `.gitignore`:

```text
.pytest_cache/
*.npz
```

(`results/`, `.venv/`, `__pycache__/`, meshes and field outputs are already ignored.)

- [ ] **Step 3: Install host dependencies**

Run: `uv pip install --python .venv/bin/python -r requirements.txt`
Expected: installs without errors.

- [ ] **Step 4: Verify the environments**

Run: `.venv/bin/python -c "import torch, SALib, sklearn, gmsh, pandas, yaml; print('host ok')"`
Expected: `host ok`

Run: `open -a Docker` (if not running), then
`docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -c "import hbsim, dolfinx, gmsh; print('container ok')"`
Expected: `container ok`

Run: `.venv/bin/python -m pytest`
Expected: `no tests ran` (exit code 5 is fine at this point).

- [ ] **Step 5: Commit**

```bash
git add -A legacy hbsim tests pytest.ini requirements.txt .gitignore
git commit -m "chore: move epoxy scripts to legacy, scaffold hbsim package and host env"
```

---

### Task 2: Materials and parameter schema

**Files:**
- Create: `hbsim/materials.py`, `hbsim/params.py`
- Test: `tests/test_materials.py`, `tests/test_params.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `materials.T_REF: float`, `materials.SI / CU / DIELECTRIC: dict`
  - `materials.to_model(p: dict) -> dict` (keys `T_head, T_chuck, v_wave, L, t_die, t_wafer, t_bond, E_d, alpha_d, phi_cu, alpha_chuck, bow_h, pressure, k_head, x0, y0` in model units)
  - `materials.bond_layer(m: dict) -> dict` (keys `E, nu, alpha, k, rho_cp`)
  - `materials.curvature(m: dict) -> float`
  - `materials.analytic_baseline(p: dict) -> {"m_x", "m_y", "d_max"}` (ppm, ppm, nm)
  - `params.ParamSpec(name, unit, low, high, default, scale)`; `params.SWEPT: dict[str, ParamSpec]` (16 entries, fixed order); `params.NUMERICS: dict`
  - `params.to_unit(name, value, spec=None) -> float`, `params.from_unit(name, u, spec=None) -> float`
  - `params.defaults() -> dict`, `params.validate(p: dict) -> None` (raises `ValueError`), `params.run_id(p: dict) -> str` (12 hex chars)
  - `params.Study` dataclass (`name, sampler, workers, timeout_s, threads_per_worker, image, specs, pinned, numerics, raw`), `params.study_from_dict(d) -> Study`, `params.load_study(path) -> Study`

- [ ] **Step 1: Write the failing tests**

`tests/test_materials.py`:

```python
import math

from hbsim import materials as M
from hbsim import params as P


def test_bond_layer_limits():
    m = M.to_model(P.defaults())
    pure_d = M.bond_layer({**m, "phi_cu": 0.0})
    assert math.isclose(pure_d["E"], m["E_d"]) and math.isclose(pure_d["alpha"], m["alpha_d"])
    pure_cu = M.bond_layer({**m, "phi_cu": 1.0})
    assert math.isclose(pure_cu["alpha"], M.CU["alpha"])


def test_turner_cte_lies_between_constituents():
    m = M.to_model(P.defaults())
    a = M.bond_layer(m)["alpha"]
    assert m["alpha_d"] < a < M.CU["alpha"]


def test_analytic_baseline_thermal_and_bow_terms():
    p = {**P.defaults(), "T_head": 60.0, "T_chuck": 40.0, "alpha_chuck": 2.6, "bow_h": 0.0}
    assert math.isclose(M.analytic_baseline(p)["m_x"], 2.6 * 20.0, rel_tol=1e-12)
    q = {**P.defaults(), "T_head": 20.0, "T_chuck": 20.0, "bow_h": 5.0, "L_die": 10.0,
         "t_die": 100.0, "t_bond": 0.2}
    kappa = 8 * 5e-3 / 10.0 ** 2
    assert math.isclose(M.analytic_baseline(q)["m_x"], kappa * (0.1 + 0.0002) / 2 * 1e6, rel_tol=1e-12)
    b = M.analytic_baseline(q)
    assert math.isclose(b["d_max"], abs(b["m_x"]) * 1e-6 * 5.0 * math.sqrt(2) * 1e6, rel_tol=1e-12)
```

`tests/test_params.py`:

```python
import math

import pytest

from hbsim import params as P


def test_unit_round_trip_linear_and_log():
    for name in ("T_head", "v_wave", "t_bond", "k_head"):
        s = P.SWEPT[name]
        for v in (s.low, s.default, s.high):
            assert math.isclose(P.from_unit(name, P.to_unit(name, v)), v, rel_tol=1e-12)


def test_log_scale_midpoint_is_geometric_mean():
    s = P.SWEPT["k_head"]
    assert math.isclose(P.from_unit("k_head", 0.5), math.sqrt(s.low * s.high), rel_tol=1e-12)


def test_defaults_validate():
    P.validate(P.defaults())


@pytest.mark.parametrize("change, message", [
    ({"T_head": 150.0}, "outside"),
    ({"t_die": 30.0, "t_bond": 5.0}, "outside"),
    ({"bow_h": float("nan")}, "finite"),
    ({"mesh_n": 21}, "even"),
    ({"n_t": 0}, "positive"),
])
def test_validate_rejects_bad_values(change, message):
    with pytest.raises(ValueError, match=message):
        P.validate({**P.defaults(), **change})


def test_validate_rejects_missing_param():
    p = P.defaults()
    del p["k_head"]
    with pytest.raises(ValueError, match="missing"):
        P.validate(p)


def test_run_id_stable_and_order_independent():
    p = P.defaults()
    q = dict(reversed(list(p.items())))
    assert P.run_id(p) == P.run_id(q)
    assert len(P.run_id(p)) == 12
    assert P.run_id({**p, "T_head": 61.0}) != P.run_id(p)


def test_study_pins_narrows_and_rejects_unknown():
    st = P.study_from_dict({"name": "s", "sampler": {"type": "lhs", "n": 4, "seed": 1},
                            "params": {"t_bond": {"value": 0.2}, "T_head": {"range": [30, 50]}}})
    assert st.pinned == {"t_bond": 0.2}
    assert "t_bond" not in st.specs
    assert (st.specs["T_head"].low, st.specs["T_head"].high) == (30.0, 50.0)
    with pytest.raises(ValueError, match="unknown params"):
        P.study_from_dict({"name": "s", "sampler": {}, "params": {"T_hed": {"value": 1}}})
    with pytest.raises(ValueError, match="within schema"):
        P.study_from_dict({"name": "s", "sampler": {}, "params": {"T_head": {"range": [0, 50]}}})
    with pytest.raises(ValueError, match="missing required key"):
        P.study_from_dict({"sampler": {}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_materials.py tests/test_params.py -q`
Expected: collection errors, `ImportError: cannot import name 'materials' from 'hbsim'` (and the same for `params`).

- [ ] **Step 3: Write the implementation**

`hbsim/materials.py`:

```python
"""Fixed material constants, unit conversion, bond-layer mixing, and the analytic overlay baseline.

Pure Python: imported both on the host and inside the DOLFINx container.
Model units: length mm, stress MPa (N/mm^2), force N, time s, temperature degC, energy J.
"""
import math

T_REF = 20.0  # degC, stress-free / pad-patterning temperature

SI = {"E": 130e3, "nu": 0.28, "alpha": 2.6e-6, "k": 0.15, "rho_cp": 1.631e-3}
CU = {"E": 120e3, "nu": 0.34, "alpha": 16.7e-6, "k": 0.40, "rho_cp": 3.45e-3}
DIELECTRIC = {"nu": 0.2, "k": 1.4e-3, "rho_cp": 1.6e-3}  # E and alpha are swept


def to_model(p):
    """Convert a resolved parameter dict (user units) to model units (mm, MPa, 1/K)."""
    return {
        "T_head": p["T_head"],
        "T_chuck": p["T_chuck"],
        "v_wave": p["v_wave"],
        "L": p["L_die"],
        "t_die": p["t_die"] * 1e-3,
        "t_wafer": p["t_wafer"] * 1e-3,
        "t_bond": p["t_bond"] * 1e-3,
        "E_d": p["E_d"] * 1e3,
        "alpha_d": p["alpha_d"] * 1e-6,
        "phi_cu": p["phi_cu"],
        "alpha_chuck": p["alpha_chuck"] * 1e-6,
        "bow_h": p["bow_h"] * 1e-3,
        "pressure": p["F_bond"] / p["L_die"] ** 2,
        "k_head": p["k_head"],
        "x0": p["x0"] * p["L_die"],
        "y0": p["y0"] * p["L_die"],
    }


def bond_layer(m):
    """Homogenized dielectric + Cu bond-layer properties (rule of mixtures; Turner CTE)."""
    phi = m["phi_cu"]
    E = (1 - phi) * m["E_d"] + phi * CU["E"]
    return {
        "E": E,
        "nu": (1 - phi) * DIELECTRIC["nu"] + phi * CU["nu"],
        "alpha": ((1 - phi) * m["E_d"] * m["alpha_d"] + phi * CU["E"] * CU["alpha"]) / E,
        "k": (1 - phi) * DIELECTRIC["k"] + phi * CU["k"],
        "rho_cp": (1 - phi) * DIELECTRIC["rho_cp"] + phi * CU["rho_cp"],
    }


def curvature(m):
    """Die curvature [1/mm] from center-to-edge bow height h: kappa = 8 h / L^2."""
    return 8.0 * m["bow_h"] / m["L"] ** 2


def analytic_baseline(p):
    """Closed-form overlay estimate (spec 5.1). Returns m_x, m_y [ppm] and d_max [nm]."""
    m = to_model(p)
    m_a = (SI["alpha"] * (m["T_head"] - T_REF)
           - m["alpha_chuck"] * (m["T_chuck"] - T_REF)
           + curvature(m) * (m["t_die"] + m["t_bond"]) / 2.0)
    d_max_mm = abs(m_a) * (m["L"] / 2.0) * math.sqrt(2.0)
    return {"m_x": m_a * 1e6, "m_y": m_a * 1e6, "d_max": d_max_mm * 1e6}
```

`hbsim/params.py`:

```python
"""Parameter schema, unit-cube scaling, validation, run IDs, and study-file loading.

Container-safe: imports only the standard library (PyYAML is imported lazily in load_study,
which only runs on the host).
"""
import hashlib
import json
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    name: str
    unit: str
    low: float
    high: float
    default: float
    scale: str = "linear"  # "linear" | "log"


SWEPT = {s.name: s for s in [
    ParamSpec("T_head", "degC", 20.0, 100.0, 60.0),
    ParamSpec("T_chuck", "degC", 20.0, 100.0, 40.0),
    ParamSpec("v_wave", "mm/s", 5.0, 200.0, 30.0, "log"),
    ParamSpec("L_die", "mm", 2.0, 20.0, 10.0),
    ParamSpec("t_die", "um", 30.0, 775.0, 100.0, "log"),
    ParamSpec("t_wafer", "um", 300.0, 775.0, 775.0),
    ParamSpec("t_bond", "um", 0.05, 1.0, 0.2, "log"),
    ParamSpec("E_d", "GPa", 60.0, 200.0, 100.0),
    ParamSpec("alpha_d", "ppm/K", 0.5, 4.0, 1.5),
    ParamSpec("phi_cu", "-", 0.05, 0.4, 0.2),
    ParamSpec("alpha_chuck", "ppm/K", 0.5, 5.0, 2.6),
    ParamSpec("bow_h", "um", -5.0, 5.0, 0.0),
    ParamSpec("F_bond", "N", 1.0, 50.0, 10.0),
    ParamSpec("k_head", "MPa/mm", 1e-3, 1e4, 1.0, "log"),
    ParamSpec("x0", "L_die", 0.0, 0.3, 0.0),
    ParamSpec("y0", "L_die", 0.0, 0.3, 0.0),
]}

# Numerical settings (not swept). Final defaults come from tests/test_convergence.py.
NUMERICS = {
    "mesh_n": 20,        # elements along the die edge (even, so the die center is a node)
    "n_ring": 6,         # elements across the wafer margin ring
    "nz_die": 4,         # through-thickness elements in die Si
    "nz_wafer": 4,       # through-thickness elements in wafer Si
    "n_t": 20,           # thermal time steps over the bond-wave window
    "margin_frac": 0.5,  # wafer margin as a fraction of L_die
    "deg": 2,            # displacement element degree
    "n_pads": 21,        # pad grid is n_pads x n_pads
}


def to_unit(name, value, spec=None):
    """Map a parameter value to [0, 1] using its spec range (log space for log scale)."""
    s = spec or SWEPT[name]
    if s.scale == "log":
        return (math.log(value) - math.log(s.low)) / (math.log(s.high) - math.log(s.low))
    return (value - s.low) / (s.high - s.low)


def from_unit(name, u, spec=None):
    """Inverse of to_unit."""
    s = spec or SWEPT[name]
    if s.scale == "log":
        return float(math.exp(math.log(s.low) + u * (math.log(s.high) - math.log(s.low))))
    return float(s.low + u * (s.high - s.low))


def defaults():
    """Fully resolved parameter set at schema defaults (swept params + numerics)."""
    return {**{n: s.default for n, s in SWEPT.items()}, **NUMERICS}


def validate(p):
    """Raise ValueError if a resolved parameter set is incomplete, out of range, or unphysical."""
    missing = [n for n in list(SWEPT) + list(NUMERICS) if n not in p]
    if missing:
        raise ValueError(f"missing parameters: {missing}")
    for n, s in SWEPT.items():
        v = p[n]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{n} must be a finite number, got {v!r}")
        tol = 1e-9 * max(abs(s.low), abs(s.high), 1.0)
        if not (s.low - tol <= v <= s.high + tol):
            raise ValueError(f"{n}={v} outside [{s.low}, {s.high}] {s.unit}")
    # Geometric sanity is guaranteed by the schema ranges: t_die >= 30 * t_bond and
    # |bow_h| <= L_die / 400, so no separate cross-parameter checks are needed.
    if int(p["mesh_n"]) != p["mesh_n"] or p["mesh_n"] < 4 or p["mesh_n"] % 2:
        raise ValueError(f"mesh_n must be an even integer >= 4, got {p['mesh_n']}")
    for n in ("n_ring", "nz_die", "nz_wafer", "n_t", "n_pads", "deg"):
        if int(p[n]) != p[n] or p[n] < 1:
            raise ValueError(f"{n} must be a positive integer, got {p[n]}")
    if not (0 < p["margin_frac"] <= 2):
        raise ValueError(f"margin_frac must be in (0, 2], got {p['margin_frac']}")


def run_id(p):
    """Stable 12-hex-char ID from the canonical JSON of a resolved parameter set."""
    blob = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class Study:
    name: str
    sampler: dict
    workers: int = 4
    timeout_s: int = 600
    threads_per_worker: int = 2
    image: str = "dolfinx/dolfinx:stable"
    specs: dict = field(default_factory=dict)     # swept name -> ParamSpec (range may be narrowed)
    pinned: dict = field(default_factory=dict)    # name -> fixed value
    numerics: dict = field(default_factory=dict)  # full NUMERICS with overrides
    raw: dict = field(default_factory=dict)       # study dict as loaded


def study_from_dict(d):
    """Build a Study from a parsed YAML dict. Params not listed are swept over the schema range."""
    for key in ("name", "sampler"):
        if key not in d:
            raise ValueError(f"study is missing required key {key!r}")
    unknown = set(d.get("params", {})) - set(SWEPT)
    if unknown:
        raise ValueError(f"unknown params in study: {sorted(unknown)}")
    specs, pinned = {}, {}
    for n, s in SWEPT.items():
        entry = d.get("params", {}).get(n) or {}
        if "value" in entry:
            pinned[n] = float(entry["value"])
        else:
            lo, hi = entry.get("range", [s.low, s.high])
            if not (s.low <= lo < hi <= s.high):
                raise ValueError(f"{n} range [{lo}, {hi}] must lie within schema [{s.low}, {s.high}]")
            specs[n] = ParamSpec(n, s.unit, float(lo), float(hi), s.default, entry.get("scale", s.scale))
    bad_num = set(d.get("numerics", {})) - set(NUMERICS)
    if bad_num:
        raise ValueError(f"unknown numerics: {sorted(bad_num)}")
    return Study(
        name=d["name"],
        sampler=d["sampler"],
        workers=int(d.get("workers", 4)),
        timeout_s=int(d.get("timeout_s", 600)),
        threads_per_worker=int(d.get("threads_per_worker", 2)),
        image=d.get("image", "dolfinx/dolfinx:stable"),
        specs=specs,
        pinned=pinned,
        numerics={**NUMERICS, **d.get("numerics", {})},
        raw=d,
    )


def load_study(path):
    import yaml  # host only: the DOLFINx image has no PyYAML
    with open(path) as f:
        return study_from_dict(yaml.safe_load(f))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_materials.py tests/test_params.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hbsim/materials.py hbsim/params.py tests/test_materials.py tests/test_params.py
git commit -m "feat: parameter schema, validation, run IDs, materials and analytic baseline"
```

---

### Task 3: Samplers and study files

**Files:**
- Create: `hbsim/sampling.py`, `studies/pilot.yaml`, `studies/main_sweep.yaml`, `studies/holdout.yaml`, `studies/lines_T_head.yaml`, `studies/lines_bow_h.yaml`, `studies/lines_k_head.yaml`, `studies/lines_t_die.yaml`, `studies/lines_v_wave.yaml`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `params.Study`, `params.SWEPT`, `params.from_unit`, `params.study_from_dict`, `params.load_study`, `params.validate` (Task 2).
- Produces: `sampling.generate(study: Study) -> list[dict]` — fully resolved parameter dicts (swept + pinned + numerics). Sobol `n` must be a power of 2.

- [ ] **Step 1: Write the failing test**

`tests/test_sampling.py`:

```python
import math

import pytest

pytest.importorskip("scipy")
from hbsim import params as P  # noqa: E402
from hbsim import sampling  # noqa: E402


def _study(sampler, params=None):
    return P.study_from_dict({"name": "s", "sampler": sampler, "params": params or {}})


def test_sobol_reproducible_in_range_and_resolved():
    st = _study({"type": "sobol", "n": 16, "seed": 42}, {"t_bond": {"value": 0.2}})
    a, b = sampling.generate(st), sampling.generate(st)
    assert a == b and len(a) == 16
    for p in a:
        P.validate(p)
        assert p["t_bond"] == 0.2 and p["mesh_n"] == P.NUMERICS["mesh_n"]


def test_sobol_requires_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        sampling.generate(_study({"type": "sobol", "n": 10, "seed": 1}))


def test_lhs_seed_changes_samples():
    a = sampling.generate(_study({"type": "lhs", "n": 8, "seed": 1}))
    b = sampling.generate(_study({"type": "lhs", "n": 8, "seed": 2}))
    assert a != b


def test_line_varies_one_param_over_range():
    rows = sampling.generate(_study({"type": "line", "param": "v_wave", "n": 5}))
    vals = [r["v_wave"] for r in rows]
    assert math.isclose(vals[0], 5.0) and math.isclose(vals[-1], 200.0)
    assert math.isclose(vals[2], math.sqrt(5.0 * 200.0))  # log-scale midpoint
    assert all(r["T_head"] == P.SWEPT["T_head"].default for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sampling.py -q`
Expected: `ImportError: cannot import name 'sampling' from 'hbsim'`.

- [ ] **Step 3: Write the implementation and study files**

`hbsim/sampling.py`:

```python
"""Design-of-experiments samplers: scrambled Sobol, Latin hypercube, one-at-a-time lines."""
import numpy as np
from scipy.stats import qmc

from hbsim.params import SWEPT, from_unit


def generate(study):
    """Return a list of fully resolved parameter dicts for a Study."""
    cfg = study.sampler
    kind = cfg["type"]
    swept = list(study.specs)
    if kind in ("sobol", "lhs"):
        n, seed = int(cfg["n"]), int(cfg["seed"])
        if kind == "sobol":
            if n < 1 or n & (n - 1):
                raise ValueError(f"sobol n must be a power of 2, got {n}")
            unit = qmc.Sobol(d=len(swept), scramble=True, seed=seed).random(n)
        else:
            unit = qmc.LatinHypercube(d=len(swept), seed=seed).random(n)
        rows = [{name: from_unit(name, u[j], study.specs[name]) for j, name in enumerate(swept)}
                for u in unit]
    elif kind == "line":
        name, n = cfg["param"], int(cfg["n"])
        if name not in study.specs:
            raise ValueError(f"line param {name} must be swept (not pinned) in the study")
        base = {k: SWEPT[k].default for k in swept}
        rows = [{**base, name: from_unit(name, u, study.specs[name])} for u in np.linspace(0, 1, n)]
    else:
        raise ValueError(f"unknown sampler type {kind!r}")
    return [{**r, **study.pinned, **study.numerics} for r in rows]
```

`studies/pilot.yaml`:

```yaml
# End-to-end check and timing estimate before the main sweep.
name: pilot
sampler: {type: sobol, n: 64, seed: 11}
workers: 4
threads_per_worker: 2
timeout_s: 600
image: dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
params: {}   # every parameter swept over its schema range (see hbsim/params.py)
```

`studies/main_sweep.yaml`:

```yaml
# Surrogate training set (spec 4.3). Sobol prefixes of size 2^k stay balanced for the learning curve.
name: main_sweep
sampler: {type: sobol, n: 2048, seed: 42}
workers: 4
threads_per_worker: 2
timeout_s: 600
image: dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
params: {}
```

`studies/holdout.yaml`:

```yaml
# Independent surrogate test set: different sampler and seed, never used for training.
name: holdout
sampler: {type: lhs, n: 256, seed: 7}
workers: 4
threads_per_worker: 2
timeout_s: 600
image: dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
params: {}
```

`studies/lines_T_head.yaml` (create the other four by replacing `T_head` with `bow_h`, `k_head`, `t_die`, `v_wave` in all three places — name, comment, and `param`):

```yaml
# One-at-a-time FEM sweep of T_head (others at schema defaults) for trend figures and surrogate checks.
name: lines_T_head
sampler: {type: line, param: T_head, n: 15}
workers: 4
threads_per_worker: 2
timeout_s: 600
image: dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
params: {}
```

- [ ] **Step 4: Run tests and validate every study file**

Run: `.venv/bin/python -m pytest tests/test_sampling.py -q`
Expected: all pass.

Run:
```bash
.venv/bin/python -c "
import glob
from hbsim import params, sampling
for f in sorted(glob.glob('studies/*.yaml')):
    rows = sampling.generate(params.load_study(f)); [params.validate(r) for r in rows]; print(f, len(rows))"
```
Expected: `holdout 256`, five `lines_* 15`, `main_sweep 2048`, `pilot 64`, no errors.

- [ ] **Step 5: Commit**

```bash
git add hbsim/sampling.py tests/test_sampling.py studies
git commit -m "feat: Sobol/LHS/line samplers and study definitions"
```

---

### Task 4: Overlay post-processing

**Files:**
- Create: `hbsim/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `postprocess.OUTPUTS = ["t_x", "t_y", "theta", "m_x", "m_y", "resid_3sigma", "d_max"]`
  - `pad_grid(L: float, n: int) -> ndarray (n*n, 2)` [mm]
  - `fit_overlay(xy, d) -> {"t_x","t_y","theta","m_x","m_y","resid"}` (mm, rad, dimensionless)
  - `remove_rigid(xy, d, fit) -> ndarray (N, 2)`
  - `metrics(xy, d) -> dict[str, float]` (nm, µrad, ppm; keys = OUTPUTS)
  - `apply_placement(xy, d, dx_nm, dy_nm, dtheta_urad) -> ndarray (N, 2)`

- [ ] **Step 1: Write the failing test**

`tests/test_postprocess.py`:

```python
import numpy as np

from hbsim import postprocess as pp


def test_pad_grid_shape_and_inset():
    xy = pp.pad_grid(10.0, 21)
    assert xy.shape == (441, 2)
    assert np.isclose(xy.max(), 4.9) and np.isclose(xy.min(), -4.9)


def test_fit_recovers_synthetic_overlay_exactly():
    xy = pp.pad_grid(10.0, 11)
    tx, ty, th, mx, my = 3e-6, -2e-6, 4e-7, 5e-5, -3e-5
    d = np.column_stack([tx - th * xy[:, 1] + mx * xy[:, 0], ty + th * xy[:, 0] + my * xy[:, 1]])
    f = pp.fit_overlay(xy, d)
    for key, val in zip(("t_x", "t_y", "theta", "m_x", "m_y"), (tx, ty, th, mx, my)):
        assert np.isclose(f[key], val, rtol=1e-9, atol=1e-15)
    assert np.abs(f["resid"]).max() < 1e-15


def test_metrics_units():
    xy = pp.pad_grid(10.0, 11)
    d = np.column_stack([1e-6 + 2e-5 * xy[:, 0], 2e-5 * xy[:, 1]])  # 1 nm shift, 20 ppm
    m = pp.metrics(xy, d)
    assert np.isclose(m["t_x"], 1.0) and np.isclose(m["m_x"], 20.0) and np.isclose(m["m_y"], 20.0)
    assert m["resid_3sigma"] < 1e-6
    assert all(isinstance(v, float) for v in m.values())


def test_remove_rigid_keeps_magnification():
    xy = pp.pad_grid(10.0, 11)
    d = np.column_stack([5e-6 - 1e-6 * xy[:, 1] + 1e-5 * xy[:, 0], 1e-6 * xy[:, 0]])
    out = pp.remove_rigid(xy, d, pp.fit_overlay(xy, d))
    f = pp.fit_overlay(xy, out)
    assert abs(f["t_x"]) < 1e-15 and abs(f["theta"]) < 1e-15 and np.isclose(f["m_x"], 1e-5)


def test_apply_placement_adds_rigid_terms():
    xy = pp.pad_grid(10.0, 5)
    out = pp.apply_placement(xy, np.zeros_like(xy), 10.0, -5.0, 2.0)
    f = pp.fit_overlay(xy, out)
    assert np.isclose(f["t_x"] * 1e6, 10.0) and np.isclose(f["t_y"] * 1e6, -5.0)
    assert np.isclose(f["theta"] * 1e6, 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: `ImportError: cannot import name 'postprocess' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/postprocess.py`:

```python
"""Overlay fitting and metrics. Pure numpy (host and container)."""
import numpy as np

OUTPUTS = ["t_x", "t_y", "theta", "m_x", "m_y", "resid_3sigma", "d_max"]


def pad_grid(L, n):
    """n x n pad positions [mm] covering the die interface, inset 2% from the edges."""
    s = np.linspace(-0.49 * L, 0.49 * L, n)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def fit_overlay(xy, d):
    """Least-squares fit d = t + theta*(-y, x) + diag(m_x, m_y)*(x, y) + r.

    xy, d: (N, 2) arrays in mm. Returns dict with t_x, t_y [mm], theta [rad], m_x, m_y [-],
    and resid (N, 2) [mm].
    """
    x, y = xy[:, 0], xy[:, 1]
    N = len(x)
    A = np.zeros((2 * N, 5))
    A[:N, 0] = 1.0
    A[:N, 2] = -y
    A[:N, 3] = x
    A[N:, 1] = 1.0
    A[N:, 2] = x
    A[N:, 4] = y
    b = np.concatenate([d[:, 0], d[:, 1]])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    return {"t_x": c[0], "t_y": c[1], "theta": c[2], "m_x": c[3], "m_y": c[4],
            "resid": np.column_stack([r[:N], r[N:]])}


def remove_rigid(xy, d, fit):
    """Subtract the translation and rotation of a fit (bonder alignment) from overlay d."""
    t = np.array([fit["t_x"], fit["t_y"]])
    rot = fit["theta"] * np.column_stack([-xy[:, 1], xy[:, 0]])
    return d - t - rot


def metrics(xy, d):
    """Paper metrics from overlay vectors in mm: lengths in nm, rotation in urad, magnification in ppm."""
    f = fit_overlay(xy, d)
    return {
        "t_x": float(f["t_x"]) * 1e6,
        "t_y": float(f["t_y"]) * 1e6,
        "theta": float(f["theta"]) * 1e6,
        "m_x": float(f["m_x"]) * 1e6,
        "m_y": float(f["m_y"]) * 1e6,
        "resid_3sigma": 3.0 * float(np.std(f["resid"])) * 1e6,
        "d_max": float(np.max(np.linalg.norm(d, axis=1))) * 1e6,
    }


def apply_placement(xy, d, dx_nm, dy_nm, dtheta_urad):
    """Add bonder placement offset [nm] and rotation error [urad] to overlay d [mm]."""
    rot = dtheta_urad * 1e-6 * np.column_stack([-xy[:, 1], xy[:, 0]])
    return d + np.array([dx_nm, dy_nm]) * 1e-6 + rot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hbsim/postprocess.py tests/test_postprocess.py
git commit -m "feat: overlay fit (translation, rotation, magnification) and metrics"
```

---

### Task 5: Structured hexahedral geometry

**Files:**
- Create: `hbsim/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `materials.to_model`, `params.defaults` (Task 2).
- Produces:
  - `geometry.levels(m) -> {"die_top", "die_bond_top", "interface", "wafer_bond_bottom", "wafer_bottom"}` (z in mm)
  - `geometry.build_model(m: dict, num: dict) -> None` — builds geometry + transfinite/recombine constraints + physical volumes (1 = die body, 2 = wafer body) in the current gmsh session; caller calls `gmsh.initialize()`, `gmsh.model.mesh.generate(3)`, `gmsh.finalize()`. `num` needs keys `mesh_n, n_ring, nz_die, nz_wafer, margin_frac`.

Key fact: with the geo kernel, `geo.mesh.setTransfiniteCurve/Surface/Recombine` must be called **before** `geo.synchronize()`; setting them on `gmsh.model.mesh` after synchronizing makes extrusion produce prisms.

- [ ] **Step 1: Write the failing test**

`tests/test_geometry.py`:

```python
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
from hbsim import geometry, materials, params  # noqa: E402


def _mesh(**overrides):
    p = {**params.defaults(), **overrides}
    m = materials.to_model(p)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geometry.build_model(m, p)
        gmsh.model.mesh.generate(3)
        types, tags, _ = gmsh.model.mesh.getElements(3)
        _, coords, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    return p, m, types, tags, coords.reshape(-1, 3)


def test_all_hexahedra_with_expected_count():
    p, _, types, tags, _ = _mesh(mesh_n=8, n_ring=2, nz_die=2, nz_wafer=2)
    assert list(types) == [5]  # gmsh type 5 = 8-node hexahedron
    n_die = 8 * 8 * (1 + 2)
    n_wafer = (8 + 2 * 2) ** 2 * (1 + 2)
    assert len(tags[0]) == n_die + n_wafer


def test_key_nodes_exist_for_pins_and_layers():
    p, m, _, _, X = _mesh(mesh_n=8, n_ring=2, nz_die=2, nz_wafer=2, t_bond=0.05)
    lv = geometry.levels(m)
    L = m["L"]

    def has(pt):
        return np.any(np.all(np.isclose(X, pt, atol=1e-9), axis=1))  # gmsh places nodes to ~1e-11

    assert has((0, 0, 0)) and has((L / 2, 0, 0)) and has((0, L / 2, 0))
    for z in lv.values():
        assert np.any(np.isclose(X[:, 2], z, atol=1e-9)), z
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -q`
Expected: `ImportError: cannot import name 'geometry' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/geometry.py`:

```python
"""Gmsh model of the die + bond layers + wafer patch as a structured hexahedral mesh.

Layout (model units, mm): interface plane z = 0. Die body (bond layer + Si) occupies
[-L/2, L/2]^2 x [0, t_bond + t_die]; wafer body occupies [-W/2, W/2]^2 x [-(t_bond + t_wafer), 0]
with W = L + 2*margin. Physical volumes: 1 = die body, 2 = wafer body.
"""
import gmsh


def levels(m):
    """z coordinates of the layer interfaces."""
    tb = m["t_bond"]
    return {"die_top": tb + m["t_die"], "die_bond_top": tb, "interface": 0.0,
            "wafer_bond_bottom": -tb, "wafer_bottom": -(tb + m["t_wafer"])}


def build_model(m, num):
    """Create the gmsh geometry and mesh constraints in the current gmsh session (no meshing)."""
    L = m["L"]
    W = L + 2 * num["margin_frac"] * L
    tb, td, tw = m["t_bond"], m["t_die"], m["t_wafer"]
    geo = gmsh.model.geo
    xs = [-W / 2, -L / 2, L / 2, W / 2]
    nseg = [int(num["n_ring"]), int(num["mesh_n"]), int(num["n_ring"])]
    pts = {(i, j): geo.addPoint(xs[i], xs[j], 0.0) for i in range(4) for j in range(4)}
    hl = {(i, j): geo.addLine(pts[i, j], pts[i + 1, j]) for j in range(4) for i in range(3)}
    vl = {(i, j): geo.addLine(pts[i, j], pts[i, j + 1]) for i in range(4) for j in range(3)}
    surf = {}
    for i in range(3):
        for j in range(3):
            loop = geo.addCurveLoop([hl[i, j], vl[i + 1, j], -hl[i, j + 1], -vl[i, j]])
            surf[i, j] = geo.addPlaneSurface([loop])
    # Structured-mesh constraints must be set on the geo kernel before synchronize,
    # otherwise extrusion produces prisms instead of hexahedra.
    for (i, _), tag in hl.items():
        geo.mesh.setTransfiniteCurve(tag, nseg[i] + 1)
    for (_, j), tag in vl.items():
        geo.mesh.setTransfiniteCurve(tag, nseg[j] + 1)
    for s in surf.values():
        geo.mesh.setTransfiniteSurface(s)
        geo.mesh.setRecombine(2, s)
    wafer_vols = []
    for s in surf.values():
        out = geo.extrude([(2, s)], 0, 0, -(tb + tw), numElements=[1, int(num["nz_wafer"])],
                          heights=[tb / (tb + tw), 1.0], recombine=True)
        wafer_vols += [t for d, t in out if d == 3]
    out = geo.extrude([(2, surf[1, 1])], 0, 0, tb + td, numElements=[1, int(num["nz_die"])],
                      heights=[tb / (tb + td), 1.0], recombine=True)
    die_vols = [t for d, t in out if d == 3]
    geo.synchronize()
    gmsh.model.addPhysicalGroup(3, die_vols, 1, name="die")
    gmsh.model.addPhysicalGroup(3, wafer_vols, 2, name="wafer")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add hbsim/geometry.py tests/test_geometry.py
git commit -m "feat: parametric structured hex mesh of die, bond layers, wafer patch"
```

---

### Task 6: Thermo-mechanical FEM and analytic verification

**Files:**
- Create: `hbsim/physics.py`
- Test: `tests/test_verification.py`

**Interfaces:**
- Consumes: `geometry.build_model`, `geometry.levels` (Task 5); `materials.SI, T_REF, bond_layer, curvature, to_model` (Task 2); `postprocess.pad_grid, fit_overlay, remove_rigid` (Task 4).
- Produces: `physics.simulate(p: dict, lock: str = "wave") -> {"pads": ndarray (N,2), "d": ndarray (N,2) [mm], "n_dofs": int}`. `lock="t0"` locks all pads at contact (verification); `lock="wave"` uses bond-wave arrival times. Does not validate ranges, so verification may pass `F_bond=0`, `phi_cu=0`, `k_head=0`, negative `x0`.

DOLFINx 0.11 facts this code relies on (verified in the prototype):
- `dolfinx.io.gmsh.model_to_mesh(model, comm, rank, gdim=3)` returns a `MeshData` with `.mesh`.
- `dolfinx.mesh.create_submesh(msh, 3, cells)` returns `(submesh, EntityMap, EntityMap, ndarray)`; parent cells via `emap.sub_topology_to_topology(sub_cells, inverse=False)`.
- Transfer between parent and submesh: `f_sub.interpolate(f_parent, cells0=parent_cells, cells1=sub_cells)`.
- Dirichlet BCs on a vector sub-space need a Function on the collapsed space: `Vc, _ = V.sub(i).collapse()`, `dofs = locate_dofs_geometrical((V.sub(i), Vc), marker)`, `dirichletbc(g, dofs, V.sub(i))`.
- Rigid-mode pins: with `k_head > 0` only the 3 in-plane pins are applied (the head spring removes the out-of-plane modes); adding `u_z` pins over-constrains the die and creates a spurious 0.2 nm translation at a centered bond wave. The 3 `u_z` pins are added only for `k_head == 0`.

- [ ] **Step 1: Write the failing test**

`tests/test_verification.py`:

```python
"""Analytic verification of the FEM model (spec 3.6). Runs inside the DOLFINx container."""
import math

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from hbsim import materials, params, physics, postprocess  # noqa: E402


def _metrics(p, lock):
    r = physics.simulate(p, lock=lock)
    return r, postprocess.metrics(r["pads"], r["d"])


def test_uniform_thermal_mismatch_matches_closed_form():
    p = {**params.defaults(), "E_d": 130.0, "alpha_d": 2.6, "phi_cu": 0.0, "alpha_chuck": 2.6,
         "bow_h": 0.0, "F_bond": 0.0, "T_head": 60.0, "T_chuck": 40.0}
    _, m = _metrics(p, "t0")
    expected_ppm = 2.6 * (60.0 - 40.0)
    assert math.isclose(m["m_x"], expected_ppm, rel_tol=0.01)
    corner = 0.49 * p["L_die"] * math.sqrt(2)
    assert math.isclose(m["d_max"], expected_ppm * 1e-6 * corner * 1e6, rel_tol=0.01)


def test_free_die_bow_matches_plate_solution():
    p = {**params.defaults(), "T_head": 20.0, "T_chuck": 20.0, "bow_h": 5.0, "F_bond": 0.0, "k_head": 0.0}
    _, m = _metrics(p, "t0")
    assert math.isclose(m["m_x"], materials.analytic_baseline(p)["m_x"], rel_tol=0.01)


def test_rigid_head_flattens_bow():
    base = {**params.defaults(), "T_head": 20.0, "T_chuck": 20.0, "bow_h": 5.0, "F_bond": 0.0}
    _, free = _metrics({**base, "k_head": 0.0}, "t0")
    _, rigid = _metrics({**base, "k_head": 1e4}, "t0")
    assert abs(rigid["m_x"]) < 0.02 * abs(free["m_x"])


def test_zero_load_gives_zero_overlay():
    p = {**params.defaults(), "T_head": 20.0, "T_chuck": 20.0, "bow_h": 0.0, "F_bond": 0.0}
    r, _ = _metrics(p, "wave")
    assert np.abs(r["d"]).max() < 1e-12


def test_mirror_symmetry_of_wave_origin():
    p = {**params.defaults(), "x0": 0.2}
    _, a = _metrics(p, "wave")
    _, b = _metrics({**p, "x0": -0.2}, "wave")
    assert math.isclose(a["t_x"], -b["t_x"], rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(a["m_x"], b["m_x"], rel_tol=1e-6)


def test_bad_lock_mode_rejected():
    with pytest.raises(ValueError, match="lock"):
        physics.simulate(params.defaults(), lock="later")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider tests/test_verification.py -q`
Expected: `ImportError: cannot import name 'physics' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/physics.py`:

```python
"""Thermo-mechanical FEM of the D2W hybrid-bonding moment (runs inside the DOLFINx container).

Step 1: transient heat conduction on the full die + wafer mesh.
Step 2: quasi-static thermo-elasticity of die and wafer as separate bodies (submeshes),
        one LU factorization per body reused for every time step.
Step 3: bond-wave locking -> overlay vectors at the pad grid.
See docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md, section 3.
"""
import gmsh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh as dmesh
from dolfinx.io import gmsh as gmshio
import dolfinx.fem.petsc as fpetsc

from hbsim import geometry as geom
from hbsim import postprocess
from hbsim.materials import SI, T_REF, bond_layer, curvature, to_model

COMM = MPI.COMM_SELF  # each run is serial; parallelism comes from running many runs at once
TOL = 1e-9
NUMERIC_KEYS = ("mesh_n", "n_ring", "nz_die", "nz_wafer", "n_t", "margin_frac", "deg", "n_pads")


def build_mesh(m, num):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geom.build_model(m, num)
        gmsh.model.mesh.generate(3)
        mesh_data = gmshio.model_to_mesh(gmsh.model, COMM, 0, gdim=3)
    finally:
        gmsh.finalize()
    return mesh_data.mesh


def _layered(msh, si_value, bond_value, t_bond):
    """DG0 field: bond_value in the bond layers (cell midpoint |z| < t_bond), si_value elsewhere."""
    f = fem.Function(fem.functionspace(msh, ("DG", 0)))
    f.interpolate(lambda x: np.where(np.abs(x[2]) < t_bond, bond_value, si_value))
    return f


def _lu_solver(A):
    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    return ksp


def solve_thermal(msh, m, times):
    """Backward-Euler heat conduction. Returns (P1 space, list of T arrays, one per time)."""
    lv = geom.levels(m)
    bl = bond_layer(m)
    V = fem.functionspace(msh, ("Lagrange", 1))
    k = _layered(msh, SI["k"], bl["k"], m["t_bond"])
    rc = _layered(msh, SI["rho_cp"], bl["rho_cp"], m["t_bond"])
    T_n = fem.Function(V)
    T_n.interpolate(lambda x: np.where(x[2] > TOL, m["T_head"], m["T_chuck"]))
    fdim = msh.topology.dim - 1
    top = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["die_top"]))
    bot = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["wafer_bottom"]))
    bcs = [fem.dirichletbc(PETSc.ScalarType(m["T_head"]), fem.locate_dofs_topological(V, fdim, top), V),
           fem.dirichletbc(PETSc.ScalarType(m["T_chuck"]), fem.locate_dofs_topological(V, fdim, bot), V)]
    dt = times[1] - times[0]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(rc * u * v * ufl.dx + dt * k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(rc * T_n * v * ufl.dx)
    A = fpetsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    ksp = _lu_solver(A)
    T = fem.Function(V)
    history = [T_n.x.array.copy()]
    for _ in times[1:]:
        b = fpetsc.assemble_vector(L)
        fpetsc.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, bcs)
        ksp.solve(b, T.x.petsc_vec)
        T.x.scatter_forward()
        T_n.x.array[:] = T.x.array
        history.append(T.x.array.copy())
        b.destroy()
    ksp.destroy()
    A.destroy()
    return V, history


def _component_bc(V, comp, marker, value_fn):
    """Dirichlet BC on one displacement component at dofs whose coordinates satisfy marker."""
    Vc, _ = V.sub(comp).collapse()
    dofs = fem.locate_dofs_geometrical((V.sub(comp), Vc), marker)
    g = fem.Function(Vc)
    g.interpolate(value_fn)
    return fem.dirichletbc(g, dofs, V.sub(comp))


def _at(px, py, pz):
    return lambda x: np.isclose(x[0], px) & np.isclose(x[1], py) & np.isclose(x[2], pz)


def _zero(x):
    return np.zeros(x.shape[1])


class ElasticBody:
    """Linear thermo-elastic body on a submesh with a reusable LU factorization."""

    def __init__(self, parent, cells, m, num, which):
        self.mesh, emap, _, _ = dmesh.create_submesh(parent, parent.topology.dim, cells)
        n_sub = self.mesh.topology.index_map(3).size_local
        self.sub_cells = np.arange(n_sub, dtype=np.int32)
        self.parent_cells = emap.sub_topology_to_topology(self.sub_cells, inverse=False)
        msh = self.mesh
        lv = geom.levels(m)
        bl = bond_layer(m)
        V = fem.functionspace(msh, ("Lagrange", int(num["deg"]), (3,)))
        E = _layered(msh, SI["E"], bl["E"], m["t_bond"])
        nu = _layered(msh, SI["nu"], bl["nu"], m["t_bond"])
        alpha = _layered(msh, SI["alpha"], bl["alpha"], m["t_bond"])
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        I = ufl.Identity(3)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(e):
            return 2 * mu * e + lmbda * ufl.tr(e) * I

        self.T = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
        eig = alpha * (self.T - T_REF) * I
        x = ufl.SpatialCoordinate(msh)
        fdim = 2
        p = m["pressure"]
        if which == "die":
            # Bow: compatible (stress-free) through-thickness eigenstrain over the whole die body.
            e_bow = -curvature(m) * (x[2] - lv["die_top"] / 2.0)
            eig = eig + ufl.as_matrix([[e_bow, 0, 0], [0, e_bow, 0], [0, 0, 0]])
            # Rigid modes: in-plane ones by statically determinate point pins on the interface
            # plane. With k_head > 0 the head spring already removes the out-of-plane modes;
            # extra u_z pins would over-constrain the die and create spurious reactions.
            L = m["L"]
            bcs = [_component_bc(V, 0, _at(0, 0, 0), _zero),
                   _component_bc(V, 1, _at(0, 0, 0), _zero),
                   _component_bc(V, 1, _at(L / 2, 0, 0), _zero)]
            if m["k_head"] == 0:  # exactly free die (verification): pin the 3 out-of-plane modes
                bcs += [_component_bc(V, 2, _at(0, 0, 0), _zero),
                        _component_bc(V, 2, _at(L / 2, 0, 0), _zero),
                        _component_bc(V, 2, _at(0, L / 2, 0), _zero)]
            # Bond force: equal and opposite pressure on die top (tag 1) and bottom (tag 2).
            top = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["die_top"]))
            bot = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], 0.0))
            traction = [(top, (0.0, 0.0, -p)), (bot, (0.0, 0.0, p))]
        elif which == "wafer":
            W2 = m["L"] / 2 + num["margin_frac"] * m["L"]
            dT = m["T_chuck"] - T_REF
            a_c, a_s = m["alpha_chuck"] * dT, SI["alpha"] * dT
            zb = lv["wafer_bottom"]

            def on_bottom(x):
                return np.isclose(x[2], zb)

            def on_sides(x):  # bottom-edge nodes belong to the bottom BC
                return ((np.isclose(np.abs(x[0]), W2) | np.isclose(np.abs(x[1]), W2))
                        & (x[2] > zb + TOL))

            bcs = [_component_bc(V, 0, on_bottom, lambda x: a_c * x[0]),
                   _component_bc(V, 1, on_bottom, lambda x: a_c * x[1]),
                   _component_bc(V, 2, on_bottom, _zero),
                   _component_bc(V, 0, on_sides, lambda x: a_s * x[0]),
                   _component_bc(V, 1, on_sides, lambda x: a_s * x[1])]
            h = m["L"] / 2 + TOL
            foot = dmesh.locate_entities_boundary(
                msh, fdim, lambda x: np.isclose(x[2], 0.0) & (np.abs(x[0]) <= h) & (np.abs(x[1]) <= h))
            traction = [(foot, (0.0, 0.0, -p))]
        else:
            raise ValueError(f"which must be 'die' or 'wafer', got {which!r}")

        facets = np.concatenate([f for f, _ in traction]).astype(np.int32)
        markers = np.concatenate([np.full(len(f), i + 1, dtype=np.int32) for i, (f, _) in enumerate(traction)])
        order = np.argsort(facets)
        ft = dmesh.meshtags(msh, fdim, facets[order], markers[order])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        lhs = ufl.inner(sigma(eps(u)), eps(v)) * ufl.dx
        if which == "die":
            # Bond-head compliance: normal spring on the die top. k_head -> 0 is a free die,
            # k_head -> infinity is a rigid flat head that flattens the bow.
            lhs = lhs + m["k_head"] * u[2] * v[2] * ds(1)
        rhs = ufl.inner(sigma(eig), eps(v)) * ufl.dx
        for i, (_, t) in enumerate(traction):
            rhs = rhs + ufl.dot(fem.Constant(msh, np.array(t, dtype=PETSc.ScalarType)), v) * ds(i + 1)
        self.a = fem.form(lhs)
        self.L = fem.form(rhs)
        self.bcs = bcs
        self.A = fpetsc.assemble_matrix(self.a, bcs=bcs)
        self.A.assemble()
        self.ksp = _lu_solver(self.A)
        self.u = fem.Function(V)
        self.V = V
        self._tree = geometry.bb_tree(msh, 3)

    def solve(self, T_parent=None, T_uniform=None):
        """Solve with a parent-mesh P1 temperature Function, or a uniform temperature."""
        if T_uniform is not None:
            self.T.x.array[:] = T_uniform
        else:
            self.T.interpolate(T_parent, cells0=self.parent_cells, cells1=self.sub_cells)
        b = fpetsc.assemble_vector(self.L)
        fpetsc.apply_lifting(b, [self.a], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, self.bcs)
        self.ksp.solve(b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        b.destroy()
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"elastic solve failed, KSP reason {reason}")

    def eval_xy(self, xy):
        """In-plane displacement (N, 2) [mm] at interface points (x, y, 0)."""
        P = np.column_stack([xy, np.zeros(len(xy))])
        coll = geometry.compute_collisions_points(self._tree, P)
        cc = geometry.compute_colliding_cells(self.mesh, coll, P)
        cells = []
        for i in range(len(P)):
            links = cc.links(i)
            if len(links) == 0:
                raise RuntimeError(f"pad point {P[i]} not found in mesh")
            cells.append(links[0])
        return self.u.eval(P, np.array(cells, dtype=np.int32))[:, :2]

    @property
    def n_dofs(self):
        return self.V.dofmap.index_map.size_global * 3

    def destroy(self):
        self.ksp.destroy()
        self.A.destroy()


def simulate(p, lock="wave"):
    """Run one simulation from a resolved parameter dict (user units). Does not validate ranges.

    lock="wave": pads lock at bond-wave arrival times (production).
    lock="t0":   all pads lock at contact, t = 0 (verification: uniform initial temperatures).
    Returns dict with pads (N,2) [mm], d (N,2) overlay [mm] after bonder alignment, n_dofs.
    """
    if lock not in ("wave", "t0"):
        raise ValueError(f"lock must be 'wave' or 't0', got {lock!r}")
    m = to_model(p)
    num = {k: p[k] for k in NUMERIC_KEYS}
    msh = build_mesh(m, num)
    pads = postprocess.pad_grid(m["L"], int(num["n_pads"]))
    die = ElasticBody(msh, dmesh.locate_entities(msh, 3, lambda x: x[2] >= -TOL), m, num, "die")
    waf = ElasticBody(msh, dmesh.locate_entities(msh, 3, lambda x: x[2] <= TOL), m, num, "wafer")
    try:
        # t = 0: pre-contact state, each body at its own uniform temperature.
        die.solve(T_uniform=m["T_head"])
        waf.solve(T_uniform=m["T_chuck"])
        d0 = die.eval_xy(pads) - waf.eval_xy(pads)
        align = postprocess.fit_overlay(pads, d0)  # bonder alignment removes this rigid motion
        if lock == "t0":
            d = postprocess.remove_rigid(pads, d0, align)
        else:
            origin = np.array([m["x0"], m["y0"]])
            corners = np.array([[sx * m["L"] / 2, sy * m["L"] / 2] for sx in (-1, 1) for sy in (-1, 1)])
            t_end = np.max(np.linalg.norm(corners - origin, axis=1)) / m["v_wave"]
            times = np.linspace(0.0, t_end, int(num["n_t"]) + 1)
            Vt, T_hist = solve_thermal(msh, m, times)
            T_par = fem.Function(Vt)
            diffs = [d0]
            for Tk in T_hist[1:]:
                T_par.x.array[:] = Tk
                die.solve(T_parent=T_par)
                waf.solve(T_parent=T_par)
                diffs.append(die.eval_xy(pads) - waf.eval_xy(pads))
            diffs = np.array(diffs)  # (n_t+1, N, 2)
            t_lock = np.linalg.norm(pads - origin, axis=1) / m["v_wave"]
            s = np.clip(t_lock / (times[1] - times[0]), 0, len(times) - 1)
            k0 = np.minimum(np.floor(s).astype(int), len(times) - 2)
            w = (s - k0)[:, None]
            idx = np.arange(len(pads))
            d_lock = (1 - w) * diffs[k0, idx] + w * diffs[k0 + 1, idx]
            d = postprocess.remove_rigid(pads, d_lock, align)
        if not np.all(np.isfinite(d)):
            raise RuntimeError("non-finite overlay values")
        return {"pads": pads, "d": d, "n_dofs": die.n_dofs + waf.n_dofs}
    finally:
        die.destroy()
        waf.destroy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider tests/test_verification.py -q`
Expected: 6 passed (about 2–3 minutes). Prototype values for reference: uniform case `m_x` = 52.0000 ppm (expected 52.0); free-die bow `m_x` = 20.06 ppm (expected 20.04); zero load max |d| ≈ 1e-17 mm; `x0 = ±0.2` gives `t_x = ±0.004687` nm.

- [ ] **Step 5: Commit**

```bash
git add hbsim/physics.py tests/test_verification.py
git commit -m "feat: thermo-elastic bonding FEM with bond-wave locking, analytic verification"
```

---

### Task 7: Single-run executor

**Files:**
- Create: `hbsim/run_one.py`
- Test: `tests/test_run_one.py`

**Interfaces:**
- Consumes: `params.validate` (Task 2), `postprocess.metrics` (Task 4), `physics.simulate` (Task 6).
- Produces: `python3 -m hbsim.run_one <run_dir>` and `run_one.execute(run_dir) -> dict`. Reads `params.json`; always writes `metrics.json` with keys `status ("ok"|"failed"), error, runtime_s, n_dofs` plus the 7 OUTPUTS on success; writes `overlay.npz` (`pads`, `d`) and `run.log`. Exit code 0 only on success. Never raises.

- [ ] **Step 1: Write the failing test**

`tests/test_run_one.py`:

```python
"""run_one writes results for good and bad parameter files. Runs inside the DOLFINx container."""
import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from hbsim import params, run_one  # noqa: E402

COARSE = {"mesh_n": 8, "n_ring": 2, "nz_die": 2, "nz_wafer": 2, "n_t": 4, "n_pads": 5}


def test_successful_run_writes_all_outputs(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({**params.defaults(), **COARSE}))
    res = run_one.execute(tmp_path)
    assert res["status"] == "ok", res["error"]
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "ok" and saved["n_dofs"] > 0 and np.isfinite(saved["m_x"])
    assert np.load(tmp_path / "overlay.npz")["d"].shape == (25, 2)
    assert (tmp_path / "run.log").read_text()


def test_invalid_params_recorded_as_failure(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({**params.defaults(), "T_head": 500.0}))
    res = run_one.execute(tmp_path)
    assert res["status"] == "failed" and "outside" in res["error"]
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "failed"


def test_missing_params_file_recorded_as_failure(tmp_path):
    res = run_one.execute(tmp_path)
    assert res["status"] == "failed" and "FileNotFoundError" in res["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider tests/test_run_one.py -q`
Expected: `ImportError: cannot import name 'run_one' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/run_one.py`:

```python
"""Execute one simulation run directory (inside the DOLFINx container).

Usage: python3 -m hbsim.run_one <run_dir>
Reads <run_dir>/params.json; writes metrics.json, overlay.npz, run.log. Exit code 0 on success.
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from hbsim import params as P
from hbsim import postprocess


def execute(run_dir):
    run_dir = Path(run_dir)
    logging.basicConfig(filename=run_dir / "run.log", level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_one")
    t0 = time.time()
    result = {"status": "failed", "error": None, "runtime_s": None, "n_dofs": None}
    try:
        p = json.loads((run_dir / "params.json").read_text())
        P.validate(p)
        log.info("params: %s", p)
        from hbsim import physics  # deferred: validation errors stay cheap and DOLFINx-free
        sim = physics.simulate(p)
        result.update(postprocess.metrics(sim["pads"], sim["d"]))
        np.savez(run_dir / "overlay.npz", pads=sim["pads"], d=sim["d"])
        result.update(status="ok", n_dofs=int(sim["n_dofs"]))
    except Exception as exc:  # every failure is recorded in metrics.json, never raised
        result["error"] = f"{type(exc).__name__}: {exc}"
        log.error("run failed:\n%s", traceback.format_exc())
    result["runtime_s"] = round(time.time() - t0, 2)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    sys.exit(0 if execute(sys.argv[1])["status"] == "ok" else 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider tests/test_run_one.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hbsim/run_one.py tests/test_run_one.py
git commit -m "feat: single-run executor writing metrics, overlay, and log"
```

---

### Task 8: Study runner and run/collect CLI

**Files:**
- Create: `hbsim/runner.py`, `cli.py` (run and collect commands only)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `params.validate, run_id, Study, study_from_dict, load_study` (Task 2), `sampling.generate` (Task 3), `hbsim.run_one` module entry point (Task 7).
- Produces:
  - `runner.PROJECT_ROOT: Path`
  - `runner.check_docker(image) -> None` (raises `RuntimeError` with an actionable message)
  - `runner.docker_executor(study) -> Callable[[Path], CompletedProcess]`
  - `runner.read_status(run_dir) -> "ok" | "failed" | None`
  - `runner.run_study(study, results_root=None, retry_failed=False, executor=None, log=print) -> Path` — writes `runs/<run_id>/params.json`, `run_ids.json`, `manifest.json`, `dataset.parquet`
  - `runner.collect(study_dir) -> DataFrame` (only runs in `run_ids.json`; missing/corrupt metrics → `status = "pending"`)
  - `cli.py run <study.yaml> [--retry-failed]`, `cli.py collect <study_dir>`

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:

```python
"""Runner orchestration with a fake executor (no Docker needed)."""
import json
import subprocess

import pytest

pytest.importorskip("pandas")
from hbsim import params, runner  # noqa: E402


def _study(n=4, **extra):
    return params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": n, "seed": 3},
                                   "workers": 2, **extra})


def _fake_ok(calls):
    def run(run_dir):
        calls.append(run_dir.name)
        p = json.loads((run_dir / "params.json").read_text())
        (run_dir / "metrics.json").write_text(json.dumps(
            {"status": "ok", "error": None, "runtime_s": 0.1, "n_dofs": 1,
             "t_x": 0.0, "t_y": 0.0, "theta": 0.0, "m_x": p["T_head"], "m_y": 0.0,
             "resid_3sigma": 0.0, "d_max": 1.0}))
        return subprocess.CompletedProcess([], 0, "", "")
    return run


def test_runs_all_then_resumes_without_rerunning(tmp_path):
    calls = []
    study_dir = runner.run_study(_study(), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 4
    df = runner.collect(study_dir)
    assert len(df) == 4 and set(df["status"]) == {"ok"}
    assert (study_dir / "dataset.parquet").exists()
    manifest = json.loads((study_dir / "manifest.json").read_text())
    assert manifest["seed"] == 3 and manifest["study"]["name"] == "t"
    runner.run_study(_study(), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 4  # nothing re-executed


def test_crash_without_metrics_is_recorded_and_retry_works(tmp_path):
    def crash(run_dir):
        return subprocess.CompletedProcess([], 137, "", "Killed")

    study_dir = runner.run_study(_study(n=2), tmp_path, executor=crash, log=lambda s: None)
    df = runner.collect(study_dir)
    assert set(df["status"]) == {"failed"} and df["error"].str.contains("137").all()
    calls = []
    runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert calls == []  # failures are skipped by default
    runner.run_study(_study(n=2), tmp_path, retry_failed=True, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 2 and set(runner.collect(study_dir)["status"]) == {"ok"}


def test_timeout_is_recorded(tmp_path):
    def slow(run_dir):
        raise subprocess.TimeoutExpired("docker", 1)

    df = runner.collect(runner.run_study(_study(n=1), tmp_path, executor=slow, log=lambda s: None))
    assert df["status"].iloc[0] == "failed" and "timeout" in df["error"].iloc[0]


def test_corrupt_metrics_file_is_treated_as_not_run(tmp_path):
    calls = []
    study_dir = runner.run_study(_study(n=1), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    run_dir = next((study_dir / "runs").iterdir())
    (run_dir / "metrics.json").write_text("{truncated")
    assert runner.collect(study_dir)["status"].iloc[0] == "pending"
    runner.run_study(_study(n=1), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 2


def test_edited_study_ignores_stale_runs(tmp_path):
    calls = []
    runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    edited = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 3, "seed": 99},
                                     "workers": 2})
    study_dir = runner.run_study(edited, tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(list((study_dir / "runs").iterdir())) == 5
    assert len(runner.collect(study_dir)) == 3


@pytest.mark.parametrize("stderr, message", [
    ("Cannot connect to the Docker daemon", "not running"),
    ("Error: No such image", "docker pull"),
])
def test_docker_preflight_messages(monkeypatch, stderr, message):
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", stderr))
    with pytest.raises(RuntimeError, match=message):
        runner.check_docker("img")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: `ImportError: cannot import name 'runner' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/runner.py`:

```python
"""Run a study: sample parameter sets, execute runs in parallel Docker containers, collect results."""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hbsim import params as P
from hbsim import sampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def docker_executor(study):
    """Return a function(run_dir) -> CompletedProcess that runs hbsim.run_one in Docker."""
    def run(run_dir):
        rel = Path(run_dir).resolve().relative_to(PROJECT_ROOT)
        cmd = ["docker", "run", "--rm", "-v", f"{PROJECT_ROOT}:/work", "-w", "/work",
               "-e", f"OMP_NUM_THREADS={study.threads_per_worker}",
               study.image, "python3", "-m", "hbsim.run_one", str(rel)]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=study.timeout_s)
    return run


def check_docker(image):
    """Fail fast with an actionable message if Docker is down or the image is missing."""
    try:
        out = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("docker command not found: install Docker Desktop") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("docker did not respond: is Docker Desktop running?") from None
    if out.returncode != 0:
        if "daemon" in out.stderr or "connect" in out.stderr:
            raise RuntimeError("Docker is not running: start Docker Desktop and retry")
        raise RuntimeError(f"Docker image {image} not found: run `docker pull {image}`")


def read_status(run_dir):
    """'ok' | 'failed' | None (not run or unreadable)."""
    f = Path(run_dir) / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("status")
    except json.JSONDecodeError:
        return None


def _record_failure(run_dir, error):
    (Path(run_dir) / "metrics.json").write_text(
        json.dumps({"status": "failed", "error": error, "runtime_s": None, "n_dofs": None}, indent=2))


def _git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _image_digest(image):
    try:
        out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run_study(study, results_root=None, retry_failed=False, executor=None, log=print):
    """Execute all pending runs of a Study and rebuild its dataset. Returns the study directory."""
    results_root = Path(results_root or PROJECT_ROOT / "results")
    study_dir = results_root / study.name
    runs_dir = study_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if executor is None:
        check_docker(study.image)
        executor = docker_executor(study)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    todo, run_ids = [], []
    for p in sampling.generate(study):
        P.validate(p)
        run_ids.append(P.run_id(p))
        run_dir = runs_dir / run_ids[-1]
        run_dir.mkdir(exist_ok=True)
        pf = run_dir / "params.json"
        if not pf.exists():
            pf.write_text(json.dumps(p, indent=2, sort_keys=True))
        status = read_status(run_dir)
        if status == "ok" or (status == "failed" and not retry_failed):
            continue
        todo.append(run_dir)
    # The current sample set defines the dataset; runs left over from an edited study are ignored.
    (study_dir / "run_ids.json").write_text(json.dumps(run_ids))
    log(f"[{study.name}] {len(todo)} runs to execute ({study.workers} workers)")

    def one(run_dir):
        (run_dir / "metrics.json").unlink(missing_ok=True)  # clear a previous failure on retry
        try:
            proc = executor(run_dir)
        except subprocess.TimeoutExpired:
            _record_failure(run_dir, f"timeout after {study.timeout_s} s")
            return run_dir, "failed"
        status = read_status(run_dir)
        if status is None:  # container died before writing metrics.json
            _record_failure(run_dir, f"exit code {proc.returncode}: {(proc.stderr or '')[-2000:]}")
            status = "failed"
        return run_dir, status

    done = 0
    with ThreadPoolExecutor(max_workers=study.workers) as pool:
        for fut in as_completed([pool.submit(one, d) for d in todo]):
            run_dir, status = fut.result()
            done += 1
            log(f"[{study.name}] {done}/{len(todo)} {run_dir.name} {status}")

    commit, dirty = _git_info()
    manifest = {"study": study.raw, "seed": study.sampler.get("seed"), "git_commit": commit,
                "git_dirty": dirty, "image": study.image, "image_digest": _image_digest(study.image),
                "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    collect(study_dir)
    return study_dir


def collect(study_dir):
    """Rebuild dataset.parquet from runs/*/params.json + metrics.json. Returns the DataFrame.

    Only runs listed in run_ids.json (the study's current sample set) are included.
    """
    study_dir = Path(study_dir)
    ids_file = study_dir / "run_ids.json"
    current = set(json.loads(ids_file.read_text())) if ids_file.exists() else None
    rows = []
    for run_dir in sorted((study_dir / "runs").iterdir()):
        if current is not None and run_dir.name not in current:
            continue
        pf, mf = run_dir / "params.json", run_dir / "metrics.json"
        if not pf.exists():
            continue
        row = {"run_id": run_dir.name, **json.loads(pf.read_text())}
        status = read_status(run_dir)
        row.update(json.loads(mf.read_text()) if status else {"status": "pending"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(study_dir / "dataset.parquet", index=False)
    return df
```

`cli.py` (first version; Task 11 replaces it with the full version):

```python
#!/usr/bin/env python3
"""Command-line entry point for the hybrid-bonding die-shift study.

  python cli.py run studies/pilot.yaml [--retry-failed]
  python cli.py collect results/pilot
"""
import argparse


def cmd_run(a):
    from hbsim import params, runner
    runner.run_study(params.load_study(a.study), retry_failed=a.retry_failed)


def cmd_collect(a):
    from hbsim import runner
    df = runner.collect(a.study_dir)
    print(df["status"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("run", help="execute a study YAML")
    p.add_argument("study")
    p.add_argument("--retry-failed", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect", help="rebuild dataset.parquet for a study directory")
    p.add_argument("study_dir")
    p.set_defaults(fn=cmd_collect)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then a real two-run smoke study through Docker**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: all pass.

Run:
```bash
cat > /tmp/smoke.yaml <<'EOF'
name: smoke
sampler: {type: lhs, n: 2, seed: 1}
workers: 2
image: dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8
numerics: {mesh_n: 8, n_ring: 2, nz_die: 2, nz_wafer: 2, n_t: 4}
EOF
.venv/bin/python cli.py run /tmp/smoke.yaml && .venv/bin/python cli.py collect results/smoke
```
Expected: two `ok` lines (about 4 s each), then `ok    2`. `results/smoke/manifest.json` contains the image digest and git commit. Then `rm -rf results/smoke`.

- [ ] **Step 5: Commit**

```bash
git add hbsim/runner.py cli.py tests/test_runner.py
git commit -m "feat: parallel Docker study runner with resume, preflight, and dataset collection"
```

---

### Task 9: Mesh convergence, numerical defaults, and pilot study

**Files:**
- Create: `tests/test_convergence.py`
- Modify (only if convergence requires it): `hbsim/params.py` `NUMERICS` block

**Interfaces:**
- Consumes: `params.defaults, NUMERICS` (Task 2), `physics.simulate` (Task 6), `postprocess.metrics` (Task 4), `cli.py run` (Task 8).
- Produces: `results/convergence.csv` (paper appendix table); final `NUMERICS` defaults; `results/pilot/` with 64 runs and measured run time.

- [ ] **Step 1: Write the convergence test**

`tests/test_convergence.py`:

```python
"""Mesh and time-step convergence (spec section 7). Slow; runs inside the DOLFINx container.

    docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -m slow tests/test_convergence.py

Refines one numerical setting at a time from the NUMERICS defaults and requires the default
level to be within 2% of the finest level for m_x and d_max. Writes results/convergence.csv.
"""
import csv
from pathlib import Path

import pytest

pytest.importorskip("dolfinx")
from hbsim import params, physics, postprocess  # noqa: E402

CASES = {
    "default": {},
    "thin_layer": {"t_bond": 0.05, "t_die": 30.0, "L_die": 20.0, "bow_h": 5.0},
}
LEVELS = {"mesh_n": [12, 20, 28], "nz_die": [2, 4, 6], "nz_wafer": [2, 4, 6], "n_t": [10, 20, 40]}
KEYS = ("m_x", "d_max")
OUT = Path(__file__).resolve().parent.parent / "results" / "convergence.csv"


def _run(p):
    r = physics.simulate(p)
    return postprocess.metrics(r["pads"], r["d"]), r["n_dofs"]


@pytest.mark.slow
def test_default_numerics_within_2_percent_of_finest():
    rows, failures = [], []
    for case, change in CASES.items():
        base = {**params.defaults(), **change}
        for setting, levels in LEVELS.items():
            results = {}
            for level in levels:
                m, n_dofs = _run({**base, setting: level})
                results[level] = m
                rows.append({"case": case, "setting": setting, "level": level, "n_dofs": n_dofs,
                             **{k: m[k] for k in KEYS}})
            finest, default = results[levels[-1]], results[params.NUMERICS[setting]]
            for k in KEYS:
                rel = abs(default[k] - finest[k]) / max(abs(finest[k]), 1e-12)
                if rel > 0.02:
                    failures.append(f"{case}/{setting}/{k}: {rel:.2%}")
        # Bond-layer effect bound: same case with the bond layers given Si properties.
        si_layer = {**base, "E_d": 130.0, "alpha_d": 2.6, "phi_cu": 0.0}
        m, n_dofs = _run(si_layer)
        rows.append({"case": case, "setting": "bond_layer_as_si", "level": "", "n_dofs": n_dofs,
                     **{k: m[k] for k in KEYS}})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert not failures, "not converged at default numerics: " + ", ".join(failures)
```

- [ ] **Step 2: Run it**

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -m slow -p no:cacheprovider tests/test_convergence.py -q`
Expected (about 15–20 minutes): `1 passed`. Prototype run (2026-10-06, 954 s): every default level within 0.2% of the finest for `m_x` and `d_max` in both cases (e.g. default case `d_max` 307.59 nm at `mesh_n` 20 vs 307.79 nm at 28; thin-layer case 692.76 vs 692.35 nm). Bond layer replaced by Si changes `d_max` by 0.26% (default) and 0.23% (thin layer).

- [ ] **Step 3: If it fails, raise the failing setting and rerun**

For each failure `case/setting/key: x%`, change that setting in `NUMERICS` (in `hbsim/params.py`) to the next level in `LEVELS` (e.g. `mesh_n` 20 → 28), extend that setting's `LEVELS` list with one finer level (e.g. append 36) so the finest stays above the default, and rerun Step 2. Repeat until it passes. Then rerun the Task 6 verification tests.

- [ ] **Step 4: Run the pilot study**

Run: `.venv/bin/python cli.py run studies/pilot.yaml`
Expected: 64 runs, all `ok`, about 10–15 minutes with 4 workers. Then check run time and failures:

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('results/pilot/dataset.parquet')
print(df['status'].value_counts().to_string()); print(df['runtime_s'].describe().round(1).to_string())
print(df[['m_x','d_max','t_x','theta','resid_3sigma']].describe().round(3).to_string())"
```
Expected: no failures; median `runtime_s` ≤ 60. If the median exceeds 90 s, set `main_sweep.yaml` `n: 1024` and note it in the spec's §9 risks. If any run failed, read its `run.log`, fix the cause, and rerun with `--retry-failed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_convergence.py hbsim/params.py
git commit -m "test: mesh/time-step convergence study; confirm numerical defaults"
```

(`results/` is git-ignored; copy `results/convergence.csv` into the paper draft separately.)

---

### Task 10: Surrogate and sensitivity analysis

**Files:**
- Create: `hbsim/surrogate.py`, `hbsim/sensitivity.py`
- Test: `tests/test_surrogate.py`

**Interfaces:**
- Consumes: `materials.analytic_baseline` (Task 2), `params.SWEPT, to_unit, from_unit` (Task 2), `postprocess.OUTPUTS` (Task 4).
- Produces:
  - `surrogate.INPUTS: list[str]` (= `list(params.SWEPT)`)
  - `surrogate.encode(df) -> ndarray (n, 16)`, `surrogate.baseline(df) -> ndarray (n, 7)`, `surrogate.ok_rows(df) -> DataFrame`
  - `surrogate.Surrogate.fit(df, n_models=5, max_epochs=3000, patience=200, seed=0, log=print)`, `.predict(df) -> (mean (n,7), std (n,7))`, `.save(dir)`, `Surrogate.load(dir)`
  - `surrogate.GPBaseline.fit(df, seed=0)`, `.predict(df) -> (mean, None)`
  - `surrogate.evaluate(model, df) -> {output: {"r2", "rmse"}}`, `surrogate.learning_curve(df_train, df_test, sizes=..., **fit_kw) -> DataFrame`
  - `sensitivity.problem()`, `sensitivity.decode(U) -> DataFrame`, `sensitivity.sobol_indices(predict, n_base=2**14, seed=0, outputs=OUTPUTS) -> (first_total DataFrame [output, param, S1, S1_conf, ST, ST_conf, interaction], second_order DataFrame [output, param_a, param_b, S2, S2_conf])`

- [ ] **Step 1: Write the failing test**

`tests/test_surrogate.py`:

```python
import numpy as np
import pytest

pytest.importorskip("torch")
from scipy.stats import qmc  # noqa: E402

from hbsim import sensitivity, surrogate  # noqa: E402
from hbsim.postprocess import OUTPUTS  # noqa: E402

I = surrogate.INPUTS


def _fake_fem(df):
    """Synthetic outputs: analytic baseline + smooth correction with a bow x k_head interaction."""
    y = surrogate.baseline(df)
    u = surrogate.encode(df)
    corr = -10 * u[:, I.index("bow_h")] * u[:, I.index("k_head")] + 3 * np.sin(3 * u[:, 0])
    y[:, OUTPUTS.index("m_x")] += corr
    y[:, OUTPUTS.index("m_y")] += corr
    y[:, OUTPUTS.index("t_x")] += 0.5 * u[:, I.index("x0")]
    return y


def _dataset(n, seed, failed=0):
    df = sensitivity.decode(qmc.Sobol(len(I), seed=seed).random(n))
    Y = _fake_fem(df)
    for j, name in enumerate(OUTPUTS):
        df[name] = Y[:, j]
    df["status"] = "ok"
    df.loc[: failed - 1, "status"] = "failed"
    return df


def test_fit_predicts_holdout_and_excludes_failures():
    logs = []
    m = surrogate.Surrogate.fit(_dataset(512, 1, failed=5), n_models=2, max_epochs=400, patience=50,
                                log=logs.append)
    assert "507 runs (5 failed/pending excluded)" in logs[0]
    assert not any("WARNING" in line for line in logs)  # 5/512 < 2%
    ev = surrogate.evaluate(m, _dataset(128, 2))
    assert ev["m_x"]["r2"] > 0.98 and ev["t_x"]["r2"] > 0.95


def test_save_load_round_trip(tmp_path):
    df = _dataset(128, 1)
    m = surrogate.Surrogate.fit(df, n_models=2, max_epochs=50, patience=10, log=lambda s: None)
    m.save(tmp_path)
    m2 = surrogate.Surrogate.load(tmp_path)
    assert np.allclose(m.predict(df)[0], m2.predict(df)[0])


def test_too_few_runs_rejected():
    with pytest.raises(ValueError, match="at least 10"):
        surrogate.Surrogate.fit(_dataset(16, 1, failed=8), log=lambda s: None)


def test_gp_baseline_fits():
    gp = surrogate.GPBaseline.fit(_dataset(128, 1))
    assert surrogate.evaluate(gp, _dataset(64, 2))["m_x"]["r2"] > 0.9


def test_sobol_finds_planted_interaction():
    ft, so = sensitivity.sobol_indices(_fake_fem, n_base=2 ** 10, outputs=["t_x"])
    top = ft.sort_values("ST", ascending=False).iloc[0]
    assert top["param"] == "x0" and top["ST"] > 0.9
    assert set(ft.columns) >= {"S1", "ST", "interaction"}
    assert len(so) == len(I) * (len(I) - 1) // 2


def test_high_failure_rate_warns():
    logs = []
    surrogate.Surrogate.fit(_dataset(128, 1, failed=10), n_models=1, max_epochs=5, patience=2,
                            log=logs.append)
    assert any("WARNING" in line and "10/128" in line for line in logs)


def test_load_rejects_mismatched_schema(tmp_path):
    import json
    m = surrogate.Surrogate.fit(_dataset(64, 1), n_models=1, max_epochs=5, patience=2, log=lambda s: None)
    m.save(tmp_path)
    sc = json.loads((tmp_path / "scalers.json").read_text())
    sc["inputs"] = sc["inputs"][:-1]
    (tmp_path / "scalers.json").write_text(json.dumps(sc))
    with pytest.raises(ValueError, match="different parameter"):
        surrogate.Surrogate.load(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_surrogate.py -q`
Expected: `ImportError: cannot import name 'sensitivity' from 'hbsim'`.

- [ ] **Step 3: Write the implementation**

`hbsim/surrogate.py`:

```python
"""Physics-informed surrogate: analytic baseline + MLP-ensemble residual; GP comparison baseline.

Host only (PyTorch, scikit-learn). Inputs are the swept parameters scaled to [0, 1] with the
schema ranges (log-scale parameters in log space), so scaling never depends on the data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from hbsim.materials import analytic_baseline
from hbsim.params import SWEPT, to_unit
from hbsim.postprocess import OUTPUTS

INPUTS = list(SWEPT)


def encode(df):
    """(n, D) unit-cube array from a DataFrame with one column per swept parameter."""
    return np.column_stack([[to_unit(n, v) for v in df[n].to_numpy(dtype=float)] for n in INPUTS])


def baseline(df):
    """(n, len(OUTPUTS)) analytic baseline; zero for outputs without a closed form."""
    out = np.zeros((len(df), len(OUTPUTS)))
    for i, row in enumerate(df[INPUTS].to_dict("records")):
        b = analytic_baseline(row)
        for j, name in enumerate(OUTPUTS):
            out[i, j] = b.get(name, 0.0)
    return out


def ok_rows(df):
    """Successful runs only."""
    return df[df["status"] == "ok"].reset_index(drop=True)


def _mlp(d_in, d_out, width=128, depth=4):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
        d = width
    layers.append(torch.nn.Linear(d, d_out))
    return torch.nn.Sequential(*layers)


def _train_one(X, Y, seed, max_epochs, patience, lr=1e-3, val_frac=0.1):
    """Train one MLP with early stopping on a random validation split. Returns (net, best val MSE)."""
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).permutation(len(X))
    n_val = max(1, int(round(val_frac * len(X))))
    va, tr = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.tensor(X[tr], dtype=torch.float32), torch.tensor(Y[tr], dtype=torch.float32)
    Xv, Yv = torch.tensor(X[va], dtype=torch.float32), torch.tensor(Y[va], dtype=torch.float32)
    net = _mlp(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(256):
            opt.zero_grad()
            loss = torch.mean((net(Xt[b]) - Yt[b]) ** 2)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.mean((net(Xv) - Yv) ** 2).item()
        if v < best - 1e-7:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


class Surrogate:
    """Ensemble of MLPs predicting standardized residuals over the analytic baseline."""

    def __init__(self, nets, mean, std):
        self.nets, self.mean, self.std = nets, np.asarray(mean), np.asarray(std)

    @classmethod
    def fit(cls, df, n_models=5, max_epochs=3000, patience=200, seed=0, log=print):
        n_all = len(df)
        df = ok_rows(df)
        if len(df) < 10:
            raise ValueError(f"need at least 10 successful runs to train, got {len(df)}")
        log(f"[surrogate] training on {len(df)} runs ({n_all - len(df)} failed/pending excluded)")
        if n_all - len(df) > 0.02 * n_all:
            log(f"[surrogate] WARNING: {n_all - len(df)}/{n_all} runs unusable (> 2%); failures may "
                "cluster in one region of parameter space and bias the surrogate. Inspect them first.")
        X = encode(df)
        R = df[OUTPUTS].to_numpy(dtype=float) - baseline(df)
        mean, std = R.mean(axis=0), R.std(axis=0)
        std[std == 0] = 1.0
        Z = (R - mean) / std
        nets = []
        for k in range(n_models):
            net, val = _train_one(X, Z, seed + k, max_epochs, patience)
            log(f"[surrogate] model {k + 1}/{n_models} val MSE (standardized) {val:.4g}")
            nets.append(net)
        return cls(nets, mean, std)

    def predict(self, df):
        """Return (mean, std) arrays of shape (n, len(OUTPUTS)) in output units."""
        X = torch.tensor(encode(df), dtype=torch.float32)
        with torch.no_grad():
            Z = np.stack([net(X).numpy() for net in self.nets])
        R = Z * self.std + self.mean
        return baseline(df) + R.mean(axis=0), R.std(axis=0)

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save([n.state_dict() for n in self.nets], out_dir / "model.pt")
        (out_dir / "scalers.json").write_text(json.dumps(
            {"inputs": INPUTS, "outputs": OUTPUTS, "mean": self.mean.tolist(), "std": self.std.tolist()},
            indent=2))

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        sc = json.loads((out_dir / "scalers.json").read_text())
        if sc["inputs"] != INPUTS or sc["outputs"] != OUTPUTS:
            raise ValueError("saved surrogate was trained with a different parameter/output list")
        nets = []
        for state in torch.load(out_dir / "model.pt"):
            net = _mlp(len(INPUTS), len(OUTPUTS))
            net.load_state_dict(state)
            net.eval()
            nets.append(net)
        return cls(nets, sc["mean"], sc["std"])


class GPBaseline:
    """One Gaussian process per output on the same residual targets (comparison model)."""

    def __init__(self, gps):
        self.gps = gps

    @classmethod
    def fit(cls, df, seed=0):
        df = ok_rows(df)
        X = encode(df)
        R = df[OUTPUTS].to_numpy(dtype=float) - baseline(df)
        gps = []
        for j in range(len(OUTPUTS)):
            kernel = ConstantKernel() * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel()
            gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed)
            gps.append(gp.fit(X, R[:, j]))
        return cls(gps)

    def predict(self, df):
        X = encode(df)
        return baseline(df) + np.column_stack([gp.predict(X) for gp in self.gps]), None


def evaluate(model, df):
    """R^2 and RMSE per output on the successful rows of df."""
    df = ok_rows(df)
    pred, _ = model.predict(df)
    true = df[OUTPUTS].to_numpy(dtype=float)
    out = {}
    for j, name in enumerate(OUTPUTS):
        err = pred[:, j] - true[:, j]
        ss_tot = np.sum((true[:, j] - true[:, j].mean()) ** 2)
        out[name] = {"r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
                     "rmse": float(np.sqrt(np.mean(err ** 2)))}
    return out


def learning_curve(df_train, df_test, sizes=(128, 256, 512, 1024, 2048), **fit_kw):
    """Holdout metrics vs training-set size (first n rows: Sobol prefixes stay balanced)."""
    df_train = ok_rows(df_train)
    rows = []
    for n in sizes:
        if n > len(df_train):
            break
        m = Surrogate.fit(df_train.iloc[:n], **fit_kw)
        for name, s in evaluate(m, df_test).items():
            rows.append({"n_train": n, "output": name, **s})
    return pd.DataFrame(rows)
```

`hbsim/sensitivity.py`:

```python
"""Global sensitivity analysis (Sobol indices) on the surrogate with SALib. Host only."""
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from hbsim.params import from_unit
from hbsim.postprocess import OUTPUTS
from hbsim.surrogate import INPUTS


def problem():
    return {"num_vars": len(INPUTS), "names": INPUTS, "bounds": [[0.0, 1.0]] * len(INPUTS)}


def decode(U):
    """Unit-cube samples (n, D) -> DataFrame of parameters in user units."""
    return pd.DataFrame({n: [from_unit(n, u) for u in U[:, j]] for j, n in enumerate(INPUTS)})


def sobol_indices(predict, n_base=2 ** 14, seed=0, outputs=OUTPUTS):
    """Sobol S1, ST, interaction (ST - S1) and second-order S2 for each output.

    predict: function(DataFrame) -> (n, len(OUTPUTS)) array, e.g. lambda df: model.predict(df)[0].
    Returns (first_total, second_order) DataFrames.
    """
    prob = problem()
    U = sobol_sample.sample(prob, n_base, calc_second_order=True, seed=seed)
    Y = predict(decode(U))
    ft, so = [], []
    for name in outputs:
        j = OUTPUTS.index(name)
        res = sobol_analyze.analyze(prob, Y[:, j], calc_second_order=True, seed=seed)
        for i, p in enumerate(INPUTS):
            ft.append({"output": name, "param": p, "S1": res["S1"][i], "S1_conf": res["S1_conf"][i],
                       "ST": res["ST"][i], "ST_conf": res["ST_conf"][i],
                       "interaction": res["ST"][i] - res["S1"][i]})
        for a in range(len(INPUTS)):
            for b in range(a + 1, len(INPUTS)):
                so.append({"output": name, "param_a": INPUTS[a], "param_b": INPUTS[b],
                           "S2": res["S2"][a, b], "S2_conf": res["S2_conf"][a, b]})
    return pd.DataFrame(ft), pd.DataFrame(so)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_surrogate.py -q`
Expected: all pass (about 10 s).

- [ ] **Step 5: Commit**

```bash
git add hbsim/surrogate.py hbsim/sensitivity.py tests/test_surrogate.py
git commit -m "feat: physics-informed MLP-ensemble surrogate, GP baseline, Sobol sensitivity"
```

---

### Task 11: Paper figures and full CLI

**Files:**
- Create: `hbsim/figures.py`
- Modify: `cli.py` (replace with the full version below)
- Test: `tests/test_cli_figures.py`

**Interfaces:**
- Consumes: `surrogate.Surrogate, GPBaseline, evaluate, learning_curve, INPUTS, ok_rows` and `sensitivity.sobol_indices` (Task 10); `params.SWEPT, defaults` (Task 2); `postprocess.OUTPUTS` (Task 4); `runner.run_study, collect` (Task 8).
- Produces:
  - `figures.parity(model, df_test, out_path, outputs=...)`, `figures.sobol_bars(first_total, output, out_path)`, `figures.s2_heatmap(second_order, output, out_path)`, `figures.process_window(model, x_param, y_param, out_path, output="d_max", spec=100.0, n=81, base=None)`, `figures.line_check(model, df_line, param, out_path, output="d_max")`, `figures.learning_curve_plot(curve, out_path, outputs=...)`, `figures.ensure_dir(path) -> Path`
  - `cli.py train --train DIR --test DIR [--out] [--n-models 5] [--gp] [--learning-curve]` → `surrogate/model.pt, scalers.json, metrics.json[, learning_curve.csv]`
  - `cli.py sensitivity --surrogate DIR [--n-base 16384] [--seed 0]` → `sobol_first_total.csv, sobol_second_order.csv`
  - `cli.py figures --surrogate DIR --test DIR [--lines DIR ...] [--spec 100] [--out]` → PDFs in `<study>/figures/`

Figure conventions (validated colorblind-safe palette): series blue `#2a78d6` and orange `#eb6834`; single-hue blue sequential ramp; gridlines `#e1e0d9`; at most two series per chart, each named in a legend; text in text colors, never series colors; labels only on the five largest S_ij cells.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_figures.py`:

```python
"""End-to-end host pipeline on synthetic data: train -> sensitivity -> figures via the CLI."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
from scipy.stats import qmc  # noqa: E402

from hbsim import sensitivity, surrogate  # noqa: E402
from hbsim.postprocess import OUTPUTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _write_study(path, n, seed, line_param=None):
    df = sensitivity.decode(qmc.Sobol(len(surrogate.INPUTS), seed=seed).random(n))
    Y = surrogate.baseline(df)
    for j, name in enumerate(OUTPUTS):
        df[name] = Y[:, j] + 0.01 * j
    df["status"] = "ok"
    path.mkdir(parents=True)
    df.to_parquet(path / "dataset.parquet")
    sampler = {"type": "line", "param": line_param, "n": n} if line_param else {"type": "sobol"}
    (path / "manifest.json").write_text(json.dumps({"study": {"sampler": sampler}}))


def _cli(*args):
    r = subprocess.run([sys.executable, str(ROOT / "cli.py"), *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_train_sensitivity_figures(tmp_path):
    tr, te, ln = tmp_path / "train", tmp_path / "test", tmp_path / "line"
    _write_study(tr, 64, 1)
    _write_study(te, 32, 2)
    _write_study(ln, 8, 3, line_param="T_head")
    out = _cli("train", "--train", str(tr), "--test", str(te), "--n-models", "1")
    assert "m_x" in out and (tr / "surrogate" / "metrics.json").exists()
    _cli("sensitivity", "--surrogate", str(tr / "surrogate"), "--n-base", "64")
    assert (tr / "surrogate" / "sobol_second_order.csv").exists()
    _cli("figures", "--surrogate", str(tr / "surrogate"), "--test", str(te), "--lines", str(ln))
    figs = {p.name for p in (tr / "figures").iterdir()}
    assert {"parity.pdf", "sobol_m_x.pdf", "s2_d_max.pdf", "window_bow_khead.pdf", "window_Thead_bow.pdf",
            "line_T_head.pdf"} <= figs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_figures.py -q`
Expected: FAIL — the assertion shows `cli.py: error: argument {run,collect}: invalid choice: 'train'`.

- [ ] **Step 3: Write the implementation**

`hbsim/figures.py`:

```python
"""Static paper figures (PDF) from datasets, the surrogate, and Sobol results. Host only.

Colors follow a validated colorblind-safe palette: categorical slots 1-2 (blue, orange),
single-hue blue sequential ramp, recessive gridlines. At most two series per chart, each
identified by legend and label, never by color alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hbsim.params import SWEPT, defaults  # noqa: E402
from hbsim.postprocess import OUTPUTS  # noqa: E402
from hbsim.surrogate import INPUTS, ok_rows  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]
TEXT, TEXT_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
UNITS = {"t_x": "nm", "t_y": "nm", "theta": "µrad", "m_x": "ppm", "m_y": "ppm",
         "resid_3sigma": "nm", "d_max": "nm"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight",
})


def _label(name):
    unit = UNITS.get(name) or SWEPT[name].unit
    return f"{name} [{unit}]"


def parity(model, df_test, out_path, outputs=("m_x", "d_max", "t_x", "resid_3sigma")):
    """Predicted vs FEM scatter per output with the y = x line; R^2 in each title."""
    df = ok_rows(df_test)
    pred, _ = model.predict(df)
    fig, axes = plt.subplots(1, len(outputs), figsize=(3.2 * len(outputs), 3.2))
    for ax, name in zip(np.atleast_1d(axes), outputs):
        j = OUTPUTS.index(name)
        true = df[name].to_numpy(dtype=float)
        lo, hi = min(true.min(), pred[:, j].min()), max(true.max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=1, label="y = x")
        ax.scatter(true, pred[:, j], s=10, color=SERIES[0], edgecolor="white", linewidth=0.5,
                   zorder=2, label="holdout run")
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = f"{1 - np.sum((pred[:, j] - true) ** 2) / ss_tot:.3f}" if ss_tot > 0 else "n/a (constant)"
        ax.set_title(f"{name}   R² = {r2}", fontsize=9)
        ax.set_xlabel(f"FEM {_label(name)}")
        ax.set_ylabel(f"surrogate {_label(name)}")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def sobol_bars(first_total, output, out_path):
    """Horizontal stacked bars per input: first-order S1 plus interaction share (ST - S1)."""
    d = first_total[first_total["output"] == output].sort_values("ST")
    s1 = d["S1"].clip(lower=0).to_numpy()
    inter = (d["ST"] - d["S1"].clip(lower=0)).clip(lower=0).to_numpy()
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.0, 0.28 * len(d) + 1.0))
    ax.barh(y, s1, height=0.6, color=SERIES[0], edgecolor="white", linewidth=1, label="first-order Sᵢ")
    ax.barh(y, inter, left=s1, height=0.6, color=SERIES[1], edgecolor="white", linewidth=1,
            label="interaction S_Tᵢ − Sᵢ")
    ax.errorbar(d["ST"], y, xerr=d["ST_conf"], fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=2)
    ax.set_yticks(y, d["param"])
    ax.set_xlabel("Sobol index")
    ax.set_title(f"Sensitivity of {_label(output)}", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def s2_heatmap(second_order, output, out_path):
    """Symmetric heatmap of second-order Sobol indices S_ij (negative estimates shown as 0)."""
    d = second_order[second_order["output"] == output]
    n = len(INPUTS)
    M = np.zeros((n, n))
    for _, r in d.iterrows():
        a, b = INPUTS.index(r["param_a"]), INPUTS.index(r["param_b"])
        M[a, b] = M[b, a] = max(r["S2"], 0.0)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, cmap=BLUES, vmin=0)
    ax.set_xticks(range(n), INPUTS, rotation=60, ha="right")
    ax.set_yticks(range(n), INPUTS)
    ax.grid(False)
    top = d.assign(S2=d["S2"].clip(lower=0)).nlargest(5, "S2")  # label only the 5 largest pairs
    for _, r in top.iterrows():
        a, b = INPUTS.index(r["param_a"]), INPUTS.index(r["param_b"])
        for i, k in ((a, b), (b, a)):
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, k] > 0.5 * M.max() else TEXT)
    fig.colorbar(im, ax=ax, label="second-order index S_ij")
    ax.set_title(f"Pairwise interactions for {_label(output)}", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


def process_window(model, x_param, y_param, out_path, output="d_max", spec=100.0, n=81, base=None):
    """Contour map of a surrogate output over two inputs (others at defaults), with a spec line."""
    base = {**defaults(), **(base or {})}
    sx, sy = SWEPT[x_param], SWEPT[y_param]

    def axis(s):
        return np.geomspace(s.low, s.high, n) if s.scale == "log" else np.linspace(s.low, s.high, n)

    X, Y = np.meshgrid(axis(sx), axis(sy))
    df = pd.DataFrame([{**{k: base[k] for k in INPUTS}, x_param: x, y_param: y}
                       for x, y in zip(X.ravel(), Y.ravel())])
    Z = model.predict(df)[0][:, OUTPUTS.index(output)].reshape(X.shape)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    cf = ax.contourf(X, Y, Z, levels=12, cmap=BLUES)
    if Z.min() < spec < Z.max():
        ax.contour(X, Y, Z, levels=[spec], colors=[SERIES[1]], linewidths=2)
        ax.legend(handles=[Line2D([], [], color=SERIES[1], linewidth=2,
                                  label=f"spec {spec:g} {UNITS[output]}")], loc="upper left")
        ax.set_title(f"{output} process window (spec {spec:g} {UNITS[output]})", fontsize=9)
    else:
        side = "everywhere above" if Z.min() >= spec else "everywhere below"
        ax.set_title(f"{output} {side} spec {spec:g} {UNITS[output]}", fontsize=9)
    if sx.scale == "log":
        ax.set_xscale("log")
    if sy.scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel(_label(x_param))
    ax.set_ylabel(_label(y_param))
    ax.grid(False)
    fig.colorbar(cf, ax=ax, label=f"surrogate {_label(output)}")
    fig.savefig(out_path)
    plt.close(fig)


def line_check(model, df_line, param, out_path, output="d_max"):
    """FEM one-at-a-time sweep (points) against the surrogate mean and ±2σ ensemble band."""
    df = ok_rows(df_line).sort_values(param)
    mean, std = model.predict(df)
    j = OUTPUTS.index(output)
    x = df[param].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.fill_between(x, mean[:, j] - 2 * std[:, j], mean[:, j] + 2 * std[:, j], color=SERIES[0],
                    alpha=0.2, linewidth=0, label="surrogate ±2σ")
    ax.plot(x, mean[:, j], color=SERIES[0], label="surrogate mean")
    ax.scatter(x, df[output], s=24, color=SERIES[1], edgecolor="white", linewidth=1, zorder=3,
               label="FEM")
    if SWEPT[param].scale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(_label(param))
    ax.set_ylabel(_label(output))
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def learning_curve_plot(curve, out_path, outputs=("m_x", "d_max")):
    """Holdout RMSE vs number of training runs (log-log), one line per output."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for color, name in zip(SERIES, outputs):
        d = curve[curve["output"] == name]
        ax.plot(d["n_train"], d["rmse"], color=color, marker="o", markersize=5, label=_label(name))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("training runs")
    ax.set_ylabel("holdout RMSE")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
```

`cli.py` (full version):

```python
#!/usr/bin/env python3
"""Command-line entry point for the hybrid-bonding die-shift study.

  python cli.py run studies/pilot.yaml [--retry-failed]
  python cli.py collect results/pilot
  python cli.py train --train results/main_sweep --test results/holdout
  python cli.py sensitivity --surrogate results/main_sweep/surrogate
  python cli.py figures --surrogate results/main_sweep/surrogate --test results/holdout
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _dataset(study_dir):
    return pd.read_parquet(Path(study_dir) / "dataset.parquet")


def cmd_run(a):
    from hbsim import params, runner
    runner.run_study(params.load_study(a.study), retry_failed=a.retry_failed)


def cmd_collect(a):
    from hbsim import runner
    df = runner.collect(a.study_dir)
    print(df["status"].value_counts().to_string())


def cmd_train(a):
    from hbsim import surrogate
    out = Path(a.out or Path(a.train) / "surrogate")
    model = surrogate.Surrogate.fit(_dataset(a.train), n_models=a.n_models)
    model.save(out)
    report = {"surrogate": surrogate.evaluate(model, _dataset(a.test))}
    if a.gp:
        gp = surrogate.GPBaseline.fit(_dataset(a.train))
        report["gp"] = surrogate.evaluate(gp, _dataset(a.test))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    if a.learning_curve:
        surrogate.learning_curve(_dataset(a.train), _dataset(a.test), n_models=a.n_models).to_csv(
            out / "learning_curve.csv", index=False)
    for name, s in report["surrogate"].items():
        print(f"{name:14s} R2={s['r2']:.4f}  RMSE={s['rmse']:.4g}")


def cmd_sensitivity(a):
    from hbsim import sensitivity, surrogate
    model = surrogate.Surrogate.load(a.surrogate)
    ft, so = sensitivity.sobol_indices(lambda df: model.predict(df)[0], n_base=a.n_base, seed=a.seed)
    ft.to_csv(Path(a.surrogate) / "sobol_first_total.csv", index=False)
    so.to_csv(Path(a.surrogate) / "sobol_second_order.csv", index=False)
    for name in ("m_x", "d_max"):
        top = ft[ft["output"] == name].sort_values("ST", ascending=False).head(5)
        print(f"\n{name}:\n" + top[["param", "S1", "ST", "interaction"]].round(3).to_string(index=False))


def cmd_figures(a):
    from hbsim import figures, surrogate
    sdir = Path(a.surrogate)
    out = figures.ensure_dir(a.out or sdir.parent / "figures")
    model = surrogate.Surrogate.load(sdir)
    figures.parity(model, _dataset(a.test), out / "parity.pdf")
    ft_path, so_path = sdir / "sobol_first_total.csv", sdir / "sobol_second_order.csv"
    if ft_path.exists():
        ft, so = pd.read_csv(ft_path), pd.read_csv(so_path)
        for name in ("m_x", "d_max"):
            figures.sobol_bars(ft, name, out / f"sobol_{name}.pdf")
            figures.s2_heatmap(so, name, out / f"s2_{name}.pdf")
    if (sdir / "learning_curve.csv").exists():
        figures.learning_curve_plot(pd.read_csv(sdir / "learning_curve.csv"), out / "learning_curve.pdf")
    figures.process_window(model, "T_head", "bow_h", out / "window_Thead_bow.pdf", spec=a.spec)
    figures.process_window(model, "bow_h", "k_head", out / "window_bow_khead.pdf", spec=a.spec)
    for line_dir in a.lines:
        df = _dataset(line_dir)
        param = json.loads((Path(line_dir) / "manifest.json").read_text())["study"]["sampler"]["param"]
        figures.line_check(model, df, param, out / f"line_{param}.pdf")
    print(f"figures written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("run", help="execute a study YAML")
    p.add_argument("study")
    p.add_argument("--retry-failed", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect", help="rebuild dataset.parquet for a study directory")
    p.add_argument("study_dir")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("train", help="train the surrogate and score it on a holdout study")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--gp", action="store_true", help="also fit and score the GP baseline")
    p.add_argument("--learning-curve", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("sensitivity", help="Sobol indices on a trained surrogate")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--n-base", type=int, default=2 ** 14)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_sensitivity)
    p = sub.add_parser("figures", help="write paper figures")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--lines", nargs="*", default=[])
    p.add_argument("--spec", type=float, default=100.0, help="d_max spec line [nm]")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_figures)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then render and look at the figures**

Run: `.venv/bin/python -m pytest tests/test_cli_figures.py -q`
Expected: 1 passed.

Then open the PDFs written by the test's temporary run (or rerun the test body against a scratch folder) and check by eye: no overlapping labels, spec line drawn or the title says the spec is not reached, legends present on every multi-series chart, constant outputs labeled "R² = n/a (constant)".

- [ ] **Step 5: Commit**

```bash
git add hbsim/figures.py cli.py tests/test_cli_figures.py
git commit -m "feat: paper figures and train/sensitivity/figures CLI commands"
```

---

### Task 12: Documentation and full verification

**Files:**
- Create: `README.md` (new; the old one is in `legacy/`)
- Modify: `docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md` (§6 geometric-sanity sentence; record convergence result)

**Interfaces:**
- Consumes: everything above.
- Produces: user documentation; spec in sync with the code.

- [ ] **Step 1: Write the README**

`README.md`:

````markdown
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
   Give it at least 8 GB of memory (Settings → Resources) for 4 parallel runs.
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

# 3. Training and test sets (main_sweep: 2,048 runs, about 6-8 h with 4 workers)
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
````

- [ ] **Step 2: Sync the spec**

In §6, replace the "Pre-solve validation" bullet with:

```markdown
- Pre-solve validation: every parameter present, finite, and inside its declared range; numerics
  are positive integers (`mesh_n` even). Geometric sanity is guaranteed by the ranges
  (`t_die ≥ 30·t_bond`, `|bow_h| ≤ L_die/400`), so no cross-parameter checks are needed.
```

In §3.4, in the "Die BCs" bullet, replace the pin description with:

```markdown
- Die BCs: the die is otherwise free. In-plane rigid modes are removed by statically determinate
  point pins on the interface plane: (0,0,0) in x, y; (L/2,0,0) in y. With `k_head > 0` the head
  spring removes the out-of-plane modes; only for `k_head = 0` (verification) are `u_z` pins added
  at (0,0,0), (L/2,0,0), (0,L/2,0). Adding `u_z` pins together with the spring over-constrains the
  die (prototype: spurious 0.2 nm translation for a centered wave).
```

In §5.2, replace the process-window bullet's default pair with: "default pairs (`T_head`, `bow_h`)
with `T_chuck` at its default (so `T_head` sets ΔT), and (`bow_h`, `k_head`)".

In §7, append to the convergence bullet: `Result (YYYY-MM-DD): <copy the pass/fail line and any NUMERICS changes from Task 9>.`

- [ ] **Step 3: Run both full test suites**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 3 skipped (the DOLFINx tests).

Run: `docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 -m pytest -p no:cacheprovider -q`
Expected: all pass; host-only tests skipped.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md
git commit -m "docs: README for the die-shift study workflow; sync spec with implementation"
```

---

## After this plan

Not part of this plan (each is a long compute run, started by the user when ready):

```bash
.venv/bin/python cli.py run studies/main_sweep.yaml          # ~6-8 h
.venv/bin/python cli.py run studies/holdout.yaml             # ~1 h
for f in studies/lines_*.yaml; do .venv/bin/python cli.py run "$f"; done
.venv/bin/python cli.py train --train results/main_sweep --test results/holdout --gp --learning-curve
.venv/bin/python cli.py sensitivity --surrogate results/main_sweep/surrogate
.venv/bin/python cli.py figures --surrogate results/main_sweep/surrogate --test results/holdout --lines results/lines_*
```

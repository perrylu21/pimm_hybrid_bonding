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
    mem_per_run_gb: float = 3.6  # peak memory of one run (3.5 GB measured at default numerics)
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
        mem_per_run_gb=float(d.get("mem_per_run_gb", 3.6)),
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

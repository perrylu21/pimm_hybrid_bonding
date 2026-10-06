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

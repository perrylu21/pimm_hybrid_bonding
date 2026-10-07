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
    "thin_layer": {"t_bond": 0.05, "t_die": 30.0, "L_die": 20.0, "kappa": 0.1}  # 5 um bow on 20 mm,
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

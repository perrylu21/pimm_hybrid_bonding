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

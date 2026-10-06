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

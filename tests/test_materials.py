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
    p = {**P.defaults(), "dT_head": 20.0, "T_chuck": 40.0, "alpha_chuck": 2.6, "kappa": 0.0}
    assert math.isclose(M.analytic_baseline(p)["m_x"], 2.6 * 20.0, rel_tol=1e-12)
    q = {**P.defaults(), "dT_head": 0.0, "T_chuck": 20.0, "kappa": 0.4, "L_die": 10.0,
         "t_die": 100.0, "t_bond": 0.2}
    kappa = 0.4e-3  # 1/mm
    assert math.isclose(M.analytic_baseline(q)["m_x"], kappa * (0.1 + 0.0002) / 2 * 1e6, rel_tol=1e-12)
    b = M.analytic_baseline(q)
    assert math.isclose(b["d_max"], abs(b["m_x"]) * 1e-6 * 5.0 * math.sqrt(2) * 1e6, rel_tol=1e-12)


def test_to_model_derives_head_temperature_and_curvature():
    m = M.to_model({**P.defaults(), "dT_head": 25.0, "T_chuck": 30.0, "kappa": -0.3})
    assert m["T_head"] == 55.0 and m["T_chuck"] == 30.0
    assert math.isclose(M.curvature(m), -0.3e-3)  # 1/m -> 1/mm

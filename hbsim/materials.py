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

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_numbers", ROOT / "scripts" / "make_numbers.py")
mn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mn)


def _e1(rows):
    return pd.DataFrame(rows, columns=["model", "n_train", "rep", "r2_d_max"])


def test_first_n_reaching_uses_median_over_reps():
    df = _e1([("m", 32, 0, 0.96), ("m", 32, 1, 0.90), ("m", 32, 2, 0.91),   # median 0.91
              ("m", 64, 0, 0.95), ("m", 64, 1, 0.97), ("m", 64, 2, 0.94)])  # median 0.95
    assert mn.first_n_reaching(df, "m", "r2_d_max", 0.95) == 64


def test_first_n_reaching_returns_none_if_never_reached():
    df = _e1([("m", 32, 0, 0.5), ("m", 64, 0, 0.6)])
    assert mn.first_n_reaching(df, "m", "r2_d_max", 0.95) is None


def test_macro_lines_are_valid_latex():
    text = mn.render({"RzeroKL": (99.06, "{:.1f}", "R0 captured [%]"), "NdmaxOurs": (64, "{:d}", "runs")})
    assert "\\newcommand{\\RzeroKL}{99.1}  % R0 captured [%]" in text
    assert "\\newcommand{\\NdmaxOurs}{64}  % runs" in text


def test_macro_names_must_be_letters_only():
    with pytest.raises(ValueError, match="letters"):
        mn.render({"R0KL": (1.0, "{:.1f}", "")})


def test_missing_value_is_rejected():
    with pytest.raises(ValueError, match="NdmaxOurs"):
        mn.render({"NdmaxOurs": (None, "{:d}", "")})


@pytest.mark.skipif(not (ROOT / "results" / "benchmark" / "e1_learning_curve.csv").exists(),
                    reason="benchmark results not present")
def test_real_benchmark_numbers_match_verified_values():
    nums = mn.compute(ROOT / "results" / "benchmark")
    got = {k: v[1].format(v[0]) for k, v in nums.items()}
    assert got["RzeroKL"] == "99.1" and got["RzeroGeneric"] == "96.2" and got["RzeroMembrane"] == "45.6"
    assert got["NdmaxOurs"] == "64" and got["NdmaxBaseline"] == "512"
    # E2 regenerated 2026-11-10 with 3 prior subsets per prior size (was 1)
    assert got["EtwoOursFour"] == "2.5" and got["EtwoOursFourSmallPrior"] == "4.4"
    assert got["EtwoPriorAffineFour"] == "2.6" and got["EtwoPriorAffineFourSmallPrior"] == "4.4"
    assert got["EtwoOursSixteenSmallPrior"] == "1.7" and got["EtwoPolyThreeSixteen"] == "2.2"
    assert got["EtwoAffineFour"] == "8.1"
    assert got["PinnFieldPhys"] == "92" and got["PinnFieldNoPhys"] == "88"


def _conv():
    rows = []
    for case, scale in (("default", 1.0), ("thin_layer", 2.0)):
        for lvl, d in ((12, 100.0), (20, 100.05), (28, 100.1)):
            rows.append({"case": case, "setting": "mesh_n", "level": lvl, "n_dofs": 1000 * lvl,
                         "m_x": 50.0, "d_max": d * scale})
        rows.append({"case": case, "setting": "bond_layer_as_si", "level": "", "n_dofs": 20000,
                     "m_x": 50.0, "d_max": 100.05 * scale * 1.003})
    return pd.DataFrame(rows)


def test_mesh_convergence_percent_is_worst_case_coarse_vs_finest():
    # coarse 100 vs finest 100.1 -> 0.0999 %, same relative gap in both cases
    assert abs(mn.coarse_vs_finest_pct(_conv(), coarse=12) - 100 * 0.1 / 100.1) < 1e-9


@pytest.mark.skipif(not (ROOT / "results" / "convergence.csv").exists(), reason="convergence study not present")
def test_reference_model_numbers():
    nums = mn.compute_reference(ROOT)
    got = {k: v[1].format(v[0]) for k, v in nums.items()}
    assert got["NTrainRuns"] == "1024" and got["NHoldoutRuns"] == "128" and got["NPads"] == "21"
    assert got["NParams"] == "16" and got["MeshCoarse"] == "12"
    assert float(got["ConvCoarsePct"]) < 2.0

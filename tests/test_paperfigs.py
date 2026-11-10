import re

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
from hbsim import paperfigs  # noqa: E402


def _e1():
    rows = []
    for m in ("physbasis", "gp_hybrid", "analytic"):
        for n in (16, 32, 64, 256):
            rows.append({"model": m, "n_train": n, "rep": 0, "r2_d_max": 0.9123, "field_rmse_nm": 49.587})
    return pd.DataFrame(rows)


def test_summary_table_is_valid_compact_latex(tmp_path):
    paperfigs.summary_table(_e1(), tmp_path)
    tex = (tmp_path / "table_e1_summary.tex").read_text()
    body = re.sub(r"\$[^$]*\$", "", tex)              # ignore math
    assert "_" not in body                             # raw underscores break LaTeX ("Missing $")
    assert r"\cmidrule" in tex and "d_{\\max}" in tex  # grouped header with a math-mode label
    assert "0.912" in tex and "49.6" in tex            # R2 to 3 decimals, RMSE to 1 decimal
    assert tex.count(r"\\") >= 5                       # header rows + one row per model
    assert (tmp_path / "table_e1_summary.csv").exists()

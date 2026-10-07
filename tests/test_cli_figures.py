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
    _write_study(ln, 8, 3, line_param="dT_head")
    out = _cli("train", "--train", str(tr), "--test", str(te), "--n-models", "1")
    assert "m_x" in out and (tr / "surrogate" / "metrics.json").exists()
    _cli("sensitivity", "--surrogate", str(tr / "surrogate"), "--n-base", "64")
    assert (tr / "surrogate" / "sobol_second_order.csv").exists()
    _cli("figures", "--surrogate", str(tr / "surrogate"), "--test", str(te), "--lines", str(ln))
    figs = {p.name for p in (tr / "figures").iterdir()}
    assert {"parity.pdf", "sobol_m_x.pdf", "s2_d_max.pdf", "window_kappa_khead.pdf", "window_dT_kappa.pdf",
            "line_dT_head.pdf"} <= figs

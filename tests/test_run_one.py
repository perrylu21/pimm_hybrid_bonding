"""run_one writes results for good and bad parameter files. Runs inside the DOLFINx container."""
import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from hbsim import params, run_one  # noqa: E402

COARSE = {"mesh_n": 8, "n_ring": 2, "nz_die": 2, "nz_wafer": 2, "n_t": 4, "n_pads": 5}


def test_successful_run_writes_all_outputs(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({**params.defaults(), **COARSE}))
    res = run_one.execute(tmp_path)
    assert res["status"] == "ok", res["error"]
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "ok" and saved["n_dofs"] > 0 and np.isfinite(saved["m_x"])
    assert np.load(tmp_path / "overlay.npz")["d"].shape == (25, 2)
    assert (tmp_path / "run.log").read_text()


def test_invalid_params_recorded_as_failure(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({**params.defaults(), "dT_head": 500.0}))
    res = run_one.execute(tmp_path)
    assert res["status"] == "failed" and "outside" in res["error"]
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "failed"


def test_missing_params_file_recorded_as_failure(tmp_path):
    res = run_one.execute(tmp_path)
    assert res["status"] == "failed" and "FileNotFoundError" in res["error"]

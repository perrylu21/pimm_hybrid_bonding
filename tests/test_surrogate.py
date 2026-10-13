import numpy as np
import pytest

pytest.importorskip("torch")
from scipy.stats import qmc  # noqa: E402

from hbsim import sensitivity, surrogate  # noqa: E402
from hbsim.postprocess import OUTPUTS  # noqa: E402

I = surrogate.INPUTS


def _fake_fem(df):
    """Synthetic outputs: analytic baseline + smooth correction with a bow x k_head interaction."""
    y = surrogate.baseline(df)
    u = surrogate.encode(df)
    corr = -10 * u[:, I.index("kappa")] * u[:, I.index("k_head")] + 3 * np.sin(3 * u[:, 0])
    y[:, OUTPUTS.index("m_x")] += corr
    y[:, OUTPUTS.index("m_y")] += corr
    y[:, OUTPUTS.index("t_x")] += 0.5 * u[:, I.index("x0")]
    return y


def _dataset(n, seed, failed=0):
    df = sensitivity.decode(qmc.Sobol(len(I), seed=seed).random(n))
    Y = _fake_fem(df)
    for j, name in enumerate(OUTPUTS):
        df[name] = Y[:, j]
    df["status"] = "ok"
    df.loc[: failed - 1, "status"] = "failed"
    return df


def test_fit_predicts_holdout_and_excludes_failures():
    logs = []
    m = surrogate.Surrogate.fit(_dataset(512, 1, failed=5), n_models=2, max_epochs=400, patience=50,
                                log=logs.append)
    assert "507 runs (5 failed/pending excluded)" in logs[0]
    assert not any("WARNING" in line for line in logs)  # 5/512 < 2%
    ev = surrogate.evaluate(m, _dataset(128, 2))
    assert ev["m_x"]["r2"] > 0.98 and ev["t_x"]["r2"] > 0.95


def test_save_load_round_trip(tmp_path):
    df = _dataset(128, 1)
    m = surrogate.Surrogate.fit(df, n_models=2, max_epochs=50, patience=10, log=lambda s: None)
    m.save(tmp_path)
    m2 = surrogate.Surrogate.load(tmp_path)
    assert np.allclose(m.predict(df)[0], m2.predict(df)[0])


def test_too_few_runs_rejected():
    with pytest.raises(ValueError, match="at least 10"):
        surrogate.Surrogate.fit(_dataset(16, 1, failed=8), log=lambda s: None)


def test_gp_baseline_fits():
    gp = surrogate.GPBaseline.fit(_dataset(128, 1))
    assert surrogate.evaluate(gp, _dataset(64, 2))["m_x"]["r2"] > 0.9


def test_sobol_finds_planted_interaction():
    ft, so = sensitivity.sobol_indices(_fake_fem, n_base=2 ** 10, outputs=["t_x"])
    top = ft.sort_values("ST", ascending=False).iloc[0]
    assert top["param"] == "x0" and top["ST"] > 0.9
    assert set(ft.columns) >= {"S1", "ST", "interaction"}
    assert len(so) == len(I) * (len(I) - 1) // 2


def test_high_failure_rate_warns():
    logs = []
    surrogate.Surrogate.fit(_dataset(128, 1, failed=10), n_models=1, max_epochs=5, patience=2,
                            log=logs.append)
    assert any("WARNING" in line and "10/128" in line for line in logs)


def test_load_rejects_mismatched_schema(tmp_path):
    import json
    m = surrogate.Surrogate.fit(_dataset(64, 1), n_models=1, max_epochs=5, patience=2, log=lambda s: None)
    m.save(tmp_path)
    sc = json.loads((tmp_path / "scalers.json").read_text())
    sc["inputs"] = sc["inputs"][:-1]
    (tmp_path / "scalers.json").write_text(json.dumps(sc))
    with pytest.raises(ValueError, match="different parameter"):
        surrogate.Surrogate.load(tmp_path)


def test_load_rejects_changed_input_range(tmp_path, monkeypatch):
    from hbsim import params
    m = surrogate.Surrogate.fit(_dataset(64, 1), n_models=1, max_epochs=5, patience=2, log=lambda s: None)
    m.save(tmp_path)
    widened = params.ParamSpec("kappa", "1/m", -1.0, 1.0, 0.0)
    monkeypatch.setitem(params.SWEPT, "kappa", widened)
    with pytest.raises(ValueError, match="kappa"):
        surrogate.Surrogate.load(tmp_path)


def test_gp_save_load_and_model_detection(tmp_path):
    df = _dataset(64, 1)
    gp = surrogate.GPBaseline.fit(df)
    gp.save(tmp_path / "gp")
    mlp = surrogate.Surrogate.fit(df, n_models=1, max_epochs=5, patience=2, log=lambda s: None)
    mlp.save(tmp_path / "mlp")
    loaded = surrogate.load_model(tmp_path / "gp")
    assert isinstance(loaded, surrogate.GPBaseline)
    assert isinstance(surrogate.load_model(tmp_path / "mlp"), surrogate.Surrogate)
    assert np.allclose(loaded.predict(df, with_std=False)[0], gp.predict(df, with_std=False)[0])


def test_gp_predict_in_batches_matches_and_std_optional():
    df, test = _dataset(64, 1), _dataset(50, 2)
    gp = surrogate.GPBaseline.fit(df)
    whole, none = gp.predict(test, with_std=False)
    batched, std = gp.predict(test, batch=7)
    assert none is None and np.allclose(whole, batched)
    assert std.shape == whole.shape and np.all(std >= 0)


def test_gp_load_rejects_changed_input_range(tmp_path, monkeypatch):
    from hbsim import params
    surrogate.GPBaseline.fit(_dataset(64, 1)).save(tmp_path)
    monkeypatch.setitem(params.SWEPT, "kappa", params.ParamSpec("kappa", "1/m", -1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="kappa"):
        surrogate.GPBaseline.load(tmp_path)

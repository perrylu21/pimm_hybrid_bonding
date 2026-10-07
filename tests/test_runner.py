"""Runner orchestration with a fake executor (no Docker needed)."""
import json
import subprocess

import pytest

pytest.importorskip("pandas")
from hbsim import params, runner  # noqa: E402


def _study(n=4, **extra):
    return params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": n, "seed": 3},
                                   "workers": 2, **extra})


def _fake_ok(calls):
    def run(run_dir):
        calls.append(run_dir.name)
        p = json.loads((run_dir / "params.json").read_text())
        (run_dir / "metrics.json").write_text(json.dumps(
            {"status": "ok", "error": None, "runtime_s": 0.1, "n_dofs": 1,
             "t_x": 0.0, "t_y": 0.0, "theta": 0.0, "m_x": p["dT_head"], "m_y": 0.0,
             "resid_3sigma": 0.0, "d_max": 1.0}))
        return subprocess.CompletedProcess([], 0, "", "")
    return run


def test_runs_all_then_resumes_without_rerunning(tmp_path):
    calls = []
    study_dir = runner.run_study(_study(), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 4
    df = runner.collect(study_dir)
    assert len(df) == 4 and set(df["status"]) == {"ok"}
    assert (study_dir / "dataset.parquet").exists()
    manifest = json.loads((study_dir / "manifest.json").read_text())
    assert manifest["seed"] == 3 and manifest["study"]["name"] == "t"
    runner.run_study(_study(), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 4  # nothing re-executed


def test_crash_without_metrics_is_recorded_and_retry_works(tmp_path):
    def crash(run_dir):
        return subprocess.CompletedProcess([], 137, "", "Killed")

    study_dir = runner.run_study(_study(n=2), tmp_path, executor=crash, log=lambda s: None)
    df = runner.collect(study_dir)
    assert set(df["status"]) == {"failed"} and df["error"].str.contains("137").all()
    calls = []
    runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert calls == []  # failures are skipped by default
    runner.run_study(_study(n=2), tmp_path, retry_failed=True, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 2 and set(runner.collect(study_dir)["status"]) == {"ok"}


def test_timeout_is_recorded(tmp_path):
    def slow(run_dir):
        raise subprocess.TimeoutExpired("docker", 1)

    df = runner.collect(runner.run_study(_study(n=1), tmp_path, executor=slow, log=lambda s: None))
    assert df["status"].iloc[0] == "failed" and "timeout" in df["error"].iloc[0]


def test_corrupt_metrics_file_is_treated_as_not_run(tmp_path):
    calls = []
    study_dir = runner.run_study(_study(n=1), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    run_dir = next((study_dir / "runs").iterdir())
    (run_dir / "metrics.json").write_text("{truncated")
    assert runner.collect(study_dir)["status"].iloc[0] == "pending"
    runner.run_study(_study(n=1), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(calls) == 2


def test_edited_study_ignores_stale_runs(tmp_path):
    calls = []
    runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    edited = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 3, "seed": 99},
                                     "workers": 2})
    study_dir = runner.run_study(edited, tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    assert len(list((study_dir / "runs").iterdir())) == 5
    assert len(runner.collect(study_dir)) == 3


@pytest.mark.parametrize("stderr, message", [
    ("Cannot connect to the Docker daemon", "not running"),
    ("Error: No such image", "docker pull"),
])
def test_docker_preflight_messages(monkeypatch, stderr, message):
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", stderr))
    with pytest.raises(RuntimeError, match=message):
        runner.check_docker("img")


@pytest.mark.parametrize("mem_gb, per_run, workers, expected", [
    (8.3, 3.6, 4, 2),    # measured: Docker VM 8.3 GB, default mesh run peaks 3.5 GB
    (16.0, 3.6, 4, 4),   # enough memory: keep the requested count
    (2.0, 3.6, 4, 1),    # never below one worker
])
def test_workers_capped_by_docker_memory(mem_gb, per_run, workers, expected):
    st = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 1, "seed": 1},
                                 "workers": workers, "mem_per_run_gb": per_run})
    assert st.mem_per_run_gb == per_run
    assert runner.effective_workers(st, mem_gb * 1e9) == expected


def test_mem_per_run_defaults_to_measured_value():
    st = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 1, "seed": 1}})
    assert st.mem_per_run_gb == 3.6


def test_timeout_kills_the_container(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    run_dir = runner.PROJECT_ROOT / "results" / "_t" / "runs" / "abc123"
    with pytest.raises(subprocess.TimeoutExpired):
        runner.docker_executor(_study())(run_dir)
    name = calls[0][calls[0].index("--name") + 1]
    assert name == "hbsim-t-abc123"
    assert ["docker", "kill", name] in calls


def test_interrupted_run_is_left_pending(tmp_path):
    def interrupted(run_dir):
        return subprocess.CompletedProcess([], 130, "", "KeyboardInterrupt")

    df = runner.collect(runner.run_study(_study(n=2), tmp_path, executor=interrupted, log=lambda s: None))
    assert set(df["status"]) == {"pending"}


def test_ctrl_c_stops_the_sweep(tmp_path):
    calls = []

    def ctrl_c(run_dir):
        calls.append(run_dir)
        raise KeyboardInterrupt

    st = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 20, "seed": 3}, "workers": 1})
    with pytest.raises(KeyboardInterrupt):
        runner.run_study(st, tmp_path, executor=ctrl_c, log=lambda s: None)
    assert len(calls) == 1
    assert set(runner.collect(tmp_path / "t")["status"]) == {"pending"}


def test_docker_failure_mid_sweep_aborts_without_recording(tmp_path):
    calls = []

    def docker_gone(run_dir):
        calls.append(run_dir)
        return subprocess.CompletedProcess([], 125, "", "Cannot connect to the Docker daemon")

    st = params.study_from_dict({"name": "t", "sampler": {"type": "lhs", "n": 20, "seed": 3}, "workers": 1})
    with pytest.raises(RuntimeError, match="Docker"):
        runner.run_study(st, tmp_path, executor=docker_gone, log=lambda s: None)
    assert len(calls) == 1
    assert set(runner.collect(tmp_path / "t")["status"]) == {"pending"}


def test_provenance_recorded_per_run_and_mixed_commits_warned(monkeypatch, tmp_path):
    calls, logs = [], []
    monkeypatch.setattr(runner, "_git_info", lambda: ("aaa", False))
    study_dir = runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    (sorted((study_dir / "runs").iterdir())[0] / "metrics.json").unlink()  # one run left unfinished
    monkeypatch.setattr(runner, "_git_info", lambda: ("bbb", True))
    runner.run_study(_study(n=2), tmp_path, executor=_fake_ok(calls), log=lambda s: None)
    df = runner.collect(study_dir, log=logs.append)
    assert sorted(df["git_commit"].unique()) == ["aaa", "bbb"]
    assert any("WARNING" in line and "2 code versions" in line for line in logs)
    history = json.loads((study_dir / "manifest.json").read_text())["history"]
    assert [h["git_commit"] for h in history] == ["aaa", "bbb"]

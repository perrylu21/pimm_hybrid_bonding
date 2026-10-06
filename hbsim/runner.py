"""Run a study: sample parameter sets, execute runs in parallel Docker containers, collect results."""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hbsim import params as P
from hbsim import sampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def docker_executor(study):
    """Return a function(run_dir) -> CompletedProcess that runs hbsim.run_one in Docker."""
    def run(run_dir):
        rel = Path(run_dir).resolve().relative_to(PROJECT_ROOT)
        cmd = ["docker", "run", "--rm", "-v", f"{PROJECT_ROOT}:/work", "-w", "/work",
               "-e", f"OMP_NUM_THREADS={study.threads_per_worker}",
               study.image, "python3", "-m", "hbsim.run_one", str(rel)]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=study.timeout_s)
    return run


def check_docker(image):
    """Fail fast with an actionable message if Docker is down or the image is missing."""
    try:
        out = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("docker command not found: install Docker Desktop") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("docker did not respond: is Docker Desktop running?") from None
    if out.returncode != 0:
        if "daemon" in out.stderr or "connect" in out.stderr:
            raise RuntimeError("Docker is not running: start Docker Desktop and retry")
        raise RuntimeError(f"Docker image {image} not found: run `docker pull {image}`")


def read_status(run_dir):
    """'ok' | 'failed' | None (not run or unreadable)."""
    f = Path(run_dir) / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("status")
    except json.JSONDecodeError:
        return None


def _record_failure(run_dir, error):
    (Path(run_dir) / "metrics.json").write_text(
        json.dumps({"status": "failed", "error": error, "runtime_s": None, "n_dofs": None}, indent=2))


def _git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _image_digest(image):
    try:
        out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run_study(study, results_root=None, retry_failed=False, executor=None, log=print):
    """Execute all pending runs of a Study and rebuild its dataset. Returns the study directory."""
    results_root = Path(results_root or PROJECT_ROOT / "results")
    study_dir = results_root / study.name
    runs_dir = study_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if executor is None:
        check_docker(study.image)
        executor = docker_executor(study)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    todo, run_ids = [], []
    for p in sampling.generate(study):
        P.validate(p)
        run_ids.append(P.run_id(p))
        run_dir = runs_dir / run_ids[-1]
        run_dir.mkdir(exist_ok=True)
        pf = run_dir / "params.json"
        if not pf.exists():
            pf.write_text(json.dumps(p, indent=2, sort_keys=True))
        status = read_status(run_dir)
        if status == "ok" or (status == "failed" and not retry_failed):
            continue
        todo.append(run_dir)
    # The current sample set defines the dataset; runs left over from an edited study are ignored.
    (study_dir / "run_ids.json").write_text(json.dumps(run_ids))
    log(f"[{study.name}] {len(todo)} runs to execute ({study.workers} workers)")

    def one(run_dir):
        (run_dir / "metrics.json").unlink(missing_ok=True)  # clear a previous failure on retry
        try:
            proc = executor(run_dir)
        except subprocess.TimeoutExpired:
            _record_failure(run_dir, f"timeout after {study.timeout_s} s")
            return run_dir, "failed"
        status = read_status(run_dir)
        if status is None:  # container died before writing metrics.json
            _record_failure(run_dir, f"exit code {proc.returncode}: {(proc.stderr or '')[-2000:]}")
            status = "failed"
        return run_dir, status

    done = 0
    with ThreadPoolExecutor(max_workers=study.workers) as pool:
        for fut in as_completed([pool.submit(one, d) for d in todo]):
            run_dir, status = fut.result()
            done += 1
            log(f"[{study.name}] {done}/{len(todo)} {run_dir.name} {status}")

    commit, dirty = _git_info()
    manifest = {"study": study.raw, "seed": study.sampler.get("seed"), "git_commit": commit,
                "git_dirty": dirty, "image": study.image, "image_digest": _image_digest(study.image),
                "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    collect(study_dir)
    return study_dir


def collect(study_dir):
    """Rebuild dataset.parquet from runs/*/params.json + metrics.json. Returns the DataFrame.

    Only runs listed in run_ids.json (the study's current sample set) are included.
    """
    study_dir = Path(study_dir)
    ids_file = study_dir / "run_ids.json"
    current = set(json.loads(ids_file.read_text())) if ids_file.exists() else None
    rows = []
    for run_dir in sorted((study_dir / "runs").iterdir()):
        if current is not None and run_dir.name not in current:
            continue
        pf, mf = run_dir / "params.json", run_dir / "metrics.json"
        if not pf.exists():
            continue
        row = {"run_id": run_dir.name, **json.loads(pf.read_text())}
        status = read_status(run_dir)
        row.update(json.loads(mf.read_text()) if status else {"status": "pending"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(study_dir / "dataset.parquet", index=False)
    return df

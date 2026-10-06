"""Run a study: sample parameter sets, execute runs in parallel Docker containers, collect results."""
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hbsim import params as P
from hbsim import sampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERRUPTED = (130, 143)  # container stopped by SIGINT / SIGTERM
DOCKER_ERROR = 125        # `docker run` itself failed (daemon gone, bad image, ...)


def docker_executor(study):
    """Return a function(run_dir) -> CompletedProcess that runs hbsim.run_one in Docker.

    Containers are named so a timed-out run can be killed: killing only the `docker run` client
    would leave the container running, holding memory and writing into the run folder later.
    """
    def run(run_dir):
        rel = Path(run_dir).resolve().relative_to(PROJECT_ROOT)
        name = f"hbsim-{study.name}-{Path(run_dir).name}"
        cmd = ["docker", "run", "--rm", "--name", name, "-v", f"{PROJECT_ROOT}:/work", "-w", "/work",
               "-e", f"OMP_NUM_THREADS={study.threads_per_worker}",
               study.image, "python3", "-m", "hbsim.run_one", str(rel)]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=study.timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60)
            raise
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


def docker_memory_bytes():
    """Memory available to Docker containers (the Docker Desktop VM on macOS), or None."""
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True,
                             text=True, timeout=30)
        return int(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def effective_workers(study, mem_bytes):
    """Parallel runs that fit in Docker's memory (95% of it), at least 1, at most study.workers.

    Running more than fit makes the kernel kill containers (exit code 137).
    """
    if not mem_bytes:
        return study.workers
    fit = int(0.95 * mem_bytes / (study.mem_per_run_gb * 1e9))
    return max(1, min(study.workers, fit))


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
    workers = study.workers
    executor_is_docker = executor is None
    if executor is None:
        check_docker(study.image)
        executor = docker_executor(study)
        mem = docker_memory_bytes()
        workers = effective_workers(study, mem)
        if workers < study.workers:
            log(f"[{study.name}] Docker has {mem / 1e9:.1f} GB; running {workers} workers instead of "
                f"{study.workers} ({study.mem_per_run_gb} GB per run). Raise Docker Desktop's memory "
                "limit (Settings > Resources) for more parallel runs.")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    commit, dirty = _git_info()
    provenance = {"git_commit": commit, "git_dirty": dirty, "image": study.image,
                  "image_digest": _image_digest(study.image) if executor_is_docker else None}

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
    log(f"[{study.name}] {len(todo)} runs to execute ({workers} workers)")

    stop = threading.Event()  # set by a worker on interrupt / Docker failure; later runs skip

    def one(run_dir):
        if stop.is_set():
            return run_dir, "skipped"
        (run_dir / "metrics.json").unlink(missing_ok=True)  # clear a previous failure on retry
        (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        try:
            proc = executor(run_dir)
        except KeyboardInterrupt:
            stop.set()
            raise
        except subprocess.TimeoutExpired:
            _record_failure(run_dir, f"timeout after {study.timeout_s} s")
            return run_dir, "failed"
        status = read_status(run_dir)
        if status is None:
            if proc.returncode in INTERRUPTED:  # Ctrl-C / SIGTERM: leave pending for resume
                stop.set()
                return run_dir, "interrupted"
            if proc.returncode == DOCKER_ERROR:  # Docker itself failed, not the simulation
                stop.set()
                return run_dir, "docker-error"
            _record_failure(run_dir, f"exit code {proc.returncode}: {(proc.stderr or '')[-2000:]}")
            status = "failed"
        return run_dir, status

    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for fut in as_completed([pool.submit(one, d) for d in todo]):
            run_dir, status = fut.result()
            if status == "docker-error":
                raise RuntimeError(f"Docker failed while starting run {run_dir.name} (exit code "
                                   f"{DOCKER_ERROR}): check Docker Desktop is running, then rerun the "
                                   "study; finished runs are kept")
            done += 1
            log(f"[{study.name}] {done}/{len(todo)} {run_dir.name} {status}")
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)  # stop queued runs; Ctrl-C returns promptly
        raise
    finally:
        pool.shutdown(wait=True)
        _write_manifest(study_dir, study, provenance, started)
        collect(study_dir, log=log)
    return study_dir


def _write_manifest(study_dir, study, provenance, started):
    """Latest invocation at top level, plus a history entry per invocation (resumes included)."""
    f = study_dir / "manifest.json"
    history = json.loads(f.read_text()).get("history", []) if f.exists() else []
    entry = {**provenance, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest = {"study": study.raw, "seed": study.sampler.get("seed"), **entry,
                "history": history + [entry]}
    f.write_text(json.dumps(manifest, indent=2))


def collect(study_dir, log=print):
    """Rebuild dataset.parquet from runs/*/params.json + metrics.json. Returns the DataFrame.

    Only runs listed in run_ids.json (the study's current sample set) are included. Each row
    carries the git commit and image that produced it; mixing code versions triggers a warning.
    """
    study_dir = Path(study_dir)
    ids_file = study_dir / "run_ids.json"
    current = set(json.loads(ids_file.read_text())) if ids_file.exists() else None
    rows = []
    for run_dir in sorted((study_dir / "runs").iterdir()):
        if current is not None and run_dir.name not in current:
            continue
        pf, mf, vf = run_dir / "params.json", run_dir / "metrics.json", run_dir / "provenance.json"
        if not pf.exists():
            continue
        row = {"run_id": run_dir.name, **json.loads(pf.read_text())}
        if vf.exists():
            row.update(json.loads(vf.read_text()))
        status = read_status(run_dir)
        row.update(json.loads(mf.read_text()) if status else {"status": "pending"})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "git_commit" in df and "status" in df:
        versions = df.loc[df["status"] == "ok", "git_commit"].dropna().unique()
        if len(versions) > 1:
            log(f"[collect] WARNING: successful runs come from {len(versions)} code versions "
                f"({', '.join(v[:7] for v in versions)}); rerun older runs before reporting results.")
    df.to_parquet(study_dir / "dataset.parquet", index=False)
    return df

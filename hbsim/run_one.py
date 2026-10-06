"""Execute one simulation run directory (inside the DOLFINx container).

Usage: python3 -m hbsim.run_one <run_dir>
Reads <run_dir>/params.json; writes metrics.json, overlay.npz, run.log. Exit code 0 on success.
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from hbsim import params as P
from hbsim import postprocess


def execute(run_dir):
    run_dir = Path(run_dir)
    logging.basicConfig(filename=run_dir / "run.log", level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_one")
    t0 = time.time()
    result = {"status": "failed", "error": None, "runtime_s": None, "n_dofs": None}
    try:
        p = json.loads((run_dir / "params.json").read_text())
        P.validate(p)
        log.info("params: %s", p)
        from hbsim import physics  # deferred: validation errors stay cheap and DOLFINx-free
        sim = physics.simulate(p)
        result.update(postprocess.metrics(sim["pads"], sim["d"]))
        np.savez(run_dir / "overlay.npz", pads=sim["pads"], d=sim["d"])
        result.update(status="ok", n_dofs=int(sim["n_dofs"]))
    except Exception as exc:  # every failure is recorded in metrics.json, never raised
        result["error"] = f"{type(exc).__name__}: {exc}"
        log.error("run failed:\n%s", traceback.format_exc())
    result["runtime_s"] = round(time.time() - t0, 2)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    sys.exit(0 if execute(sys.argv[1])["status"] == "ok" else 1)

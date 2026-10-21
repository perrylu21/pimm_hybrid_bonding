"""Pad-level overlay fields from finished runs, in normalized die coordinates. Host only.

Every run stores the overlay at an n_pads x n_pads grid (overlay.npz: pads [mm], d [mm]). For the
field models the die is mapped to the reference square [-1, 1]^2 with xi = x / (L_die / 2), and the
overlay is stored in nm. Fields are flattened as [d_x at all pads, d_y at all pads].
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hbsim.params import SWEPT
from hbsim.postprocess import OUTPUTS, metrics, pad_grid

INPUTS = list(SWEPT)


def ref_pads(n_pads=21):
    """Pad positions on the reference square (L = 2), shape (n_pads^2, 2)."""
    return pad_grid(2.0, n_pads)


class FieldSet:
    """Parameters (DataFrame, one row per run) and overlay fields (n_runs, 2 * n_pads^2) in nm."""

    def __init__(self, df, F, n_pads=21):
        self.df = df.reset_index(drop=True)
        self.F = np.asarray(F, dtype=float)
        self.n_pads = n_pads
        self.P = ref_pads(n_pads)

    def __len__(self):
        return len(self.df)

    @property
    def half_L(self):
        """L_die / 2 per run [mm]."""
        return self.df["L_die"].to_numpy(dtype=float) / 2.0

    def subset(self, idx):
        return FieldSet(self.df.iloc[idx], self.F[idx], self.n_pads)

    def scalars(self, F=None):
        """Paper metrics (OUTPUTS) computed from fields (default: the stored fields)."""
        F = self.F if F is None else F
        return scalars_from_fields(self.df, F, self.P)


def scalars_from_fields(df, F, P):
    """OUTPUTS for each field, with the same overlay fit the FEM post-processing uses."""
    n = len(P)
    rows = []
    for L, f in zip(df["L_die"].to_numpy(dtype=float), F):
        rows.append(metrics(P * L / 2.0, np.column_stack([f[:n], f[n:]]) * 1e-6))
    return pd.DataFrame(rows, columns=OUTPUTS)


def load_fields(study_dir):
    """FieldSet of all successful runs in results/<study>/runs/*/ (sorted by run id)."""
    rows, F, n_pads = [], [], None
    for run in sorted(Path(study_dir, "runs").iterdir()):
        mfile, ofile = run / "metrics.json", run / "overlay.npz"
        if not ofile.exists():
            continue
        if mfile.exists() and json.loads(mfile.read_text()).get("status") != "ok":
            continue
        p = json.loads((run / "params.json").read_text())
        z = np.load(ofile)
        n = int(round(np.sqrt(len(z["pads"]))))
        if n_pads is None:
            n_pads = n
        if n != n_pads or not np.allclose(z["pads"], pad_grid(p["L_die"], n), atol=1e-9):
            raise ValueError(f"{run}: pad grid differs from the {n_pads} x {n_pads} reference grid")
        rows.append({**{k: p[k] for k in INPUTS}, "run_id": run.name})
        F.append(np.concatenate([z["d"][:, 0], z["d"][:, 1]]) * 1e6)
    if not rows:
        raise FileNotFoundError(f"no finished runs with overlay.npz in {study_dir}")
    return FieldSet(pd.DataFrame(rows), np.array(F), n_pads)

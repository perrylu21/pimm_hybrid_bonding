"""Physics-informed surrogate: analytic baseline + MLP-ensemble residual; GP comparison baseline.

Host only (PyTorch, scikit-learn). Inputs are the swept parameters scaled to [0, 1] with the
schema ranges (log-scale parameters in log space), so scaling never depends on the data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from hbsim.materials import analytic_baseline
from hbsim.params import SWEPT, to_unit
from hbsim.postprocess import OUTPUTS

INPUTS = list(SWEPT)


def encode(df):
    """(n, D) unit-cube array from a DataFrame with one column per swept parameter."""
    return np.column_stack([[to_unit(n, v) for v in df[n].to_numpy(dtype=float)] for n in INPUTS])


def baseline(df):
    """(n, len(OUTPUTS)) analytic baseline; zero for outputs without a closed form."""
    out = np.zeros((len(df), len(OUTPUTS)))
    for i, row in enumerate(df[INPUTS].to_dict("records")):
        b = analytic_baseline(row)
        for j, name in enumerate(OUTPUTS):
            out[i, j] = b.get(name, 0.0)
    return out


def _input_specs():
    """[name, low, high, scale] per input: the scaling encode() applies, saved with the model."""
    return [[n, SWEPT[n].low, SWEPT[n].high, SWEPT[n].scale] for n in INPUTS]


def ok_rows(df):
    """Successful runs only."""
    return df[df["status"] == "ok"].reset_index(drop=True)


def _mlp(d_in, d_out, width=128, depth=4):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
        d = width
    layers.append(torch.nn.Linear(d, d_out))
    return torch.nn.Sequential(*layers)


def _train_one(X, Y, seed, max_epochs, patience, lr=1e-3, val_frac=0.1):
    """Train one MLP with early stopping on a random validation split. Returns (net, best val MSE)."""
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).permutation(len(X))
    n_val = max(1, int(round(val_frac * len(X))))
    va, tr = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.tensor(X[tr], dtype=torch.float32), torch.tensor(Y[tr], dtype=torch.float32)
    Xv, Yv = torch.tensor(X[va], dtype=torch.float32), torch.tensor(Y[va], dtype=torch.float32)
    net = _mlp(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(256):
            opt.zero_grad()
            loss = torch.mean((net(Xt[b]) - Yt[b]) ** 2)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.mean((net(Xv) - Yv) ** 2).item()
        if v < best - 1e-7:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


class Surrogate:
    """Ensemble of MLPs predicting standardized residuals over the analytic baseline."""

    def __init__(self, nets, mean, std):
        self.nets, self.mean, self.std = nets, np.asarray(mean), np.asarray(std)

    @classmethod
    def fit(cls, df, n_models=5, max_epochs=3000, patience=200, seed=0, log=print):
        n_all = len(df)
        df = ok_rows(df)
        if len(df) < 10:
            raise ValueError(f"need at least 10 successful runs to train, got {len(df)}")
        log(f"[surrogate] training on {len(df)} runs ({n_all - len(df)} failed/pending excluded)")
        if n_all - len(df) > 0.02 * n_all:
            log(f"[surrogate] WARNING: {n_all - len(df)}/{n_all} runs unusable (> 2%); failures may "
                "cluster in one region of parameter space and bias the surrogate. Inspect them first.")
        X = encode(df)
        R = df[OUTPUTS].to_numpy(dtype=float) - baseline(df)
        mean, std = R.mean(axis=0), R.std(axis=0)
        std[std == 0] = 1.0
        Z = (R - mean) / std
        nets = []
        for k in range(n_models):
            net, val = _train_one(X, Z, seed + k, max_epochs, patience)
            log(f"[surrogate] model {k + 1}/{n_models} val MSE (standardized) {val:.4g}")
            nets.append(net)
        return cls(nets, mean, std)

    def predict(self, df):
        """Return (mean, std) arrays of shape (n, len(OUTPUTS)) in output units."""
        X = torch.tensor(encode(df), dtype=torch.float32)
        with torch.no_grad():
            Z = np.stack([net(X).numpy() for net in self.nets])
        R = Z * self.std + self.mean
        return baseline(df) + R.mean(axis=0), R.std(axis=0)

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save([n.state_dict() for n in self.nets], out_dir / "model.pt")
        (out_dir / "scalers.json").write_text(json.dumps(
            {"inputs": INPUTS, "input_specs": _input_specs(), "outputs": OUTPUTS,
             "mean": self.mean.tolist(), "std": self.std.tolist()}, indent=2))

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        sc = json.loads((out_dir / "scalers.json").read_text())
        if sc["inputs"] != INPUTS or sc["outputs"] != OUTPUTS:
            raise ValueError("saved surrogate was trained with a different parameter/output list")
        changed = [a[0] for a, b in zip(sc.get("input_specs", []), _input_specs()) if a != b]
        if "input_specs" not in sc or changed:
            raise ValueError(f"saved surrogate was trained with different input ranges/scales "
                             f"({', '.join(changed) or 'not recorded'}); retrain it")
        nets = []
        for state in torch.load(out_dir / "model.pt"):
            net = _mlp(len(INPUTS), len(OUTPUTS))
            net.load_state_dict(state)
            net.eval()
            nets.append(net)
        return cls(nets, sc["mean"], sc["std"])


class GPBaseline:
    """One Gaussian process per output on the same residual targets (comparison model)."""

    def __init__(self, gps):
        self.gps = gps

    @classmethod
    def fit(cls, df, seed=0):
        df = ok_rows(df)
        X = encode(df)
        R = df[OUTPUTS].to_numpy(dtype=float) - baseline(df)
        gps = []
        for j in range(len(OUTPUTS)):
            kernel = ConstantKernel() * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel()
            gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed)
            gps.append(gp.fit(X, R[:, j]))
        return cls(gps)

    def predict(self, df):
        X = encode(df)
        return baseline(df) + np.column_stack([gp.predict(X) for gp in self.gps]), None


def evaluate(model, df):
    """R^2 and RMSE per output on the successful rows of df."""
    df = ok_rows(df)
    pred, _ = model.predict(df)
    true = df[OUTPUTS].to_numpy(dtype=float)
    out = {}
    for j, name in enumerate(OUTPUTS):
        err = pred[:, j] - true[:, j]
        ss_tot = np.sum((true[:, j] - true[:, j].mean()) ** 2)
        out[name] = {"r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
                     "rmse": float(np.sqrt(np.mean(err ** 2)))}
    return out


def learning_curve(df_train, df_test, sizes=(128, 256, 512, 1024, 2048), **fit_kw):
    """Holdout metrics vs training-set size (first n rows: Sobol prefixes stay balanced)."""
    df_train = ok_rows(df_train)
    rows = []
    for n in sizes:
        if n > len(df_train):
            break
        m = Surrogate.fit(df_train.iloc[:n], **fit_kw)
        for name, s in evaluate(m, df_test).items():
            rows.append({"n_train": n, "output": name, **s})
    return pd.DataFrame(rows)

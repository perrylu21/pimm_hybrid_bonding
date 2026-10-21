"""Benchmarks for the data-scarce die-shift paper (experiments E1-E4). Host only.

E1  learning curves: holdout accuracy vs number of FEM training runs, several random subsets.
E2  sparse metrology: reconstruct the full pad overlay of a die from k overlay marks.
E3  robustness: noisy training data and out-of-distribution (extrapolation) tests.
E4  ablations: run as extra models inside E1 (no physics loss, no analytic prior).
R0  representation test: how much of the FEM overlay the reduced physics can represent.

Every experiment appends one CSV row per (model, setting, repeat) and skips rows already present,
so an interrupted run resumes where it stopped.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from hbsim import physbasis as pb
from hbsim.fielddata import INPUTS
from hbsim.postprocess import OUTPUTS

SCORED = ["m_x", "resid_3sigma", "d_max"]


# ------------------------------------------------------------------------------------------------
# Model registry: field models return (n, 2N) fields in nm; scalar models return OUTPUTS directly.
# ------------------------------------------------------------------------------------------------

def _scalar_df(fs):
    return pd.concat([fs.df[INPUTS].reset_index(drop=True), fs.scalars()], axis=1).assign(status="ok")


class _ScalarGP:
    """Pure data-driven GP per output, no physics or analytic prior (lower bound baseline)."""

    def fit(self, fs):
        df = _scalar_df(fs)
        X = pb.encode(df)
        self.gps = {}
        for o in OUTPUTS:
            k = ConstantKernel() * Matern(length_scale=np.ones(len(INPUTS)), nu=2.5) + WhiteKernel()
            self.gps[o] = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(X, df[o])
        return self

    def predict_scalars(self, df):
        X = pb.encode(df)
        return pd.DataFrame({o: g.predict(X) for o, g in self.gps.items()})


class _Wrapped:
    """Perry's existing surrogates (analytic baseline + GP / MLP-ensemble residual)."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, fs):
        from hbsim import surrogate
        df = _scalar_df(fs)
        if self.kind == "gp":
            self.m = surrogate.GPBaseline.fit(df, seed=self.seed)
        else:
            self.m = surrogate.Surrogate.fit(df, n_models=5, seed=self.seed, log=lambda *a: None)
        return self

    def predict_scalars(self, df):
        pred, _ = self.m.predict(df, with_std=False)
        return pd.DataFrame(pred, columns=OUTPUTS)


class _Analytic:
    def fit(self, fs):
        self.P = fs.P
        return self

    def predict(self, df):
        return pb.analytic_fields(df, self.P)


def make_model(name, seed=0, pinn_steps=6000):
    from hbsim.pinn import PINNModel
    return {
        "analytic": lambda: _Analytic(),
        "gp_data_only": lambda: _ScalarGP(),
        "gp_hybrid": lambda: _Wrapped("gp", seed),
        "mlp_hybrid": lambda: _Wrapped("mlp", seed),
        "physbasis": lambda: pb.PhysicsBasisGP(seed=seed),
        "physbasis_noprior": lambda: pb.PhysicsBasisGP(use_prior=False, seed=seed),
        "physbasis_membrane_only": lambda: pb.PhysicsBasisGP(bending=False, seed=seed),
        "pinn": lambda: PINNModel(steps=pinn_steps, seed=seed),
        "pinn_nophys": lambda: PINNModel(steps=pinn_steps, w_pde=0.0, w_bc=0.0, seed=seed),
        "pinn_noprior": lambda: PINNModel(steps=pinn_steps, use_prior=False, seed=seed),
        "pinn_nobending": lambda: PINNModel(steps=pinn_steps, bending=False, seed=seed),
    }[name]()


def predict_all(model, test):
    """(scalar DataFrame, fields or None) for any registered model."""
    if hasattr(model, "predict_scalars"):
        return model.predict_scalars(test.df), None
    F = model.predict(test.df)
    return test.scalars(F), F


def score(test, scal, F):
    true = test.scalars()
    out = {}
    for o in SCORED:
        err = scal[o].to_numpy() - true[o].to_numpy()
        out[f"r2_{o}"] = 1 - (err ** 2).sum() / ((true[o] - true[o].mean()) ** 2).sum()
        out[f"rmse_{o}"] = float(np.sqrt((err ** 2).mean()))
    if F is not None:
        n = len(test.P)
        e = F - test.F
        mag = np.sqrt(e[:, :n] ** 2 + e[:, n:] ** 2)
        out["field_rmse_nm"] = float(np.sqrt((e ** 2).mean()))
        out["field_p95_nm"] = float(np.percentile(mag, 95))
    return out


def _append(path, row):
    path = Path(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _done(path, keys):
    path = Path(path)
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[keys].astype(str).to_numpy()))


def subset_idx(n_total, n, rep):
    return np.random.default_rng(1000 * n + rep).choice(n_total, n, replace=False)


# ------------------------------------------------------------------------------------------------
# E1 / E4: learning curves (+ ablations)
# ------------------------------------------------------------------------------------------------

def run_e1(train, test, out_csv, models, sizes, reps, log=print, pinn_steps=6000):
    done = _done(out_csv, ["model", "n_train", "rep"])
    for n in sizes:
        for rep in range(reps):
            sub = train.subset(subset_idx(len(train), n, rep))
            for name in models:
                if (name, str(n), str(rep)) in done:
                    continue
                t = time.time()
                model = make_model(name, seed=rep, pinn_steps=pinn_steps).fit(sub)
                t_fit = time.time() - t
                t = time.time()
                scal, F = predict_all(model, test)
                t_pred = (time.time() - t) / len(test)
                row = {"model": name, "n_train": n, "rep": rep, "fit_s": t_fit, "predict_s_per_die": t_pred,
                       **score(test, scal, F)}
                _append(out_csv, row)
                log(f"[E1] {name:18s} N={n:5d} rep={rep}  R2 m_x {row['r2_m_x']:.3f}  "
                    f"d_max {row['r2_d_max']:.3f}  resid {row['r2_resid_3sigma']:.3f}  "
                    f"field {row.get('field_rmse_nm', float('nan')):.1f} nm  ({t_fit:.0f}s)")


# ------------------------------------------------------------------------------------------------
# E2: sparse overlay marks
# ------------------------------------------------------------------------------------------------

def _cv_score_var(train_sub, model, seed=0):
    """Out-of-fold variance of each predicted score of a PhysicsBasisGP (prior calibration)."""
    T = model.score_targets(train_sub)
    res = np.zeros_like(T)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(T):
        m = pb.PhysicsBasisGP(deg=model.deg, use_prior=model.use_prior, seed=seed)
        m.fit(train_sub.subset(tr), modes_from=model)
        res[va] = T[va] - m.predict_scores(train_sub.df.iloc[va])
    return res.var(0)


def run_e2(train, test, out_csv, n_prior=(64, 256), ks=(3, 4, 5, 9, 16), noises=(1.0, 2.0, 5.0),
           reps=1, inflate=2.0, log=print):
    P = test.P
    n = len(P)
    rig = pb.rigid_modes(P)
    done = _done(out_csv, ["n_prior", "k", "noise_nm", "rep"])
    for npr in n_prior:
        for rep in range(reps):
            sub = train.subset(subset_idx(len(train), npr, rep))
            model = pb.PhysicsBasisGP(seed=rep).fit(sub)
            var = _cv_score_var(sub, model, seed=rep) * inflate
            M = np.column_stack([rig, model.modes()])
            S = np.diag(np.r_[[1e4] * 3, var])
            prior = model.predict_norm(test.df)
            rng = np.random.default_rng(rep)
            for k in ks:
                marks = pb.mark_layout(k, test.n_pads)
                rows = np.r_[marks, marks + n]
                for noise in noises:
                    if (str(npr), str(k), str(noise), str(rep)) in done:
                        continue
                    acc = {m: [] for m in ("affine", "physics_prior_only", "physics_map")}
                    for t in range(len(test)):
                        hl = test.half_L[t]
                        meas = test.F[t, rows] + noise * rng.standard_normal(len(rows))
                        preds = {"affine": pb.reconstruct_affine(P, marks, meas) if 2 * k >= 6 else None,
                                 "physics_prior_only": prior[t] * hl,
                                 "physics_map": pb.reconstruct_physics(P, marks, meas, hl, prior[t], S, M, noise)}
                        for m, f in preds.items():
                            if f is None:
                                continue
                            e = f - test.F[t]
                            mag = np.sqrt(e[:n] ** 2 + e[n:] ** 2)
                            acc[m].append((np.sqrt((e ** 2).mean()), mag.max()))
                    for m, v in acc.items():
                        if not v:
                            continue
                        v = np.array(v)
                        row = {"n_prior": npr, "k": k, "noise_nm": noise, "rep": rep, "method": m,
                               "rmse_median_nm": float(np.median(v[:, 0])), "rmse_mean_nm": float(v[:, 0].mean()),
                               "max_err_median_nm": float(np.median(v[:, 1])),
                               "max_err_p90_nm": float(np.percentile(v[:, 1], 90))}
                        _append(out_csv, row)
                    log(f"[E2] prior N={npr} k={k:2d} noise={noise} nm  " + "  ".join(
                        f"{m} {np.median(np.array(v)[:, 0]):.2f}" for m, v in acc.items() if v))


# ------------------------------------------------------------------------------------------------
# E3: robustness
# ------------------------------------------------------------------------------------------------

def run_e3(train, test, out_csv, models, n=128, noises=(2.0, 5.0), reps=1, log=print, pinn_steps=6000):
    """Noisy training fields, and two extrapolation splits (hot bond head, large die)."""
    done = _done(out_csv, ["model", "case", "rep"])
    cases = []
    for s in noises:
        cases.append((f"train_noise_{s:g}nm", None, None, s))
    cases.append(("ood_dT_head_gt40", train.df["dT_head"] <= 40, test.df["dT_head"] > 40, 0.0))
    cases.append(("ood_L_die_gt14", train.df["L_die"] <= 14, test.df["L_die"] > 14, 0.0))
    for case, tr_mask, te_mask, noise in cases:
        for rep in range(reps):
            pool = np.arange(len(train)) if tr_mask is None else np.where(tr_mask.to_numpy())[0]
            sub = train.subset(pool[subset_idx(len(pool), min(n, len(pool)), rep)])
            if noise > 0:
                rng = np.random.default_rng(rep)
                from hbsim.fielddata import FieldSet
                sub = FieldSet(sub.df, sub.F + noise * rng.standard_normal(sub.F.shape), sub.n_pads)
            te = test if te_mask is None else test.subset(np.where(te_mask.to_numpy())[0])
            for name in models:
                if (name, case, str(rep)) in done:
                    continue
                t = time.time()
                model = make_model(name, seed=rep, pinn_steps=pinn_steps).fit(sub)
                scal, F = predict_all(model, te)
                row = {"model": name, "case": case, "rep": rep, "n_train": len(sub), "n_test": len(te),
                       "fit_s": time.time() - t, **score(te, scal, F)}
                _append(out_csv, row)
                log(f"[E3] {case:20s} {name:18s} R2 m_x {row['r2_m_x']:.3f} d_max {row['r2_d_max']:.3f} "
                    f"resid {row['r2_resid_3sigma']:.3f}")


# ------------------------------------------------------------------------------------------------
# R0: representation test of the reduced physics
# ------------------------------------------------------------------------------------------------

def run_r0(train, out_csv, degs=(2, 3, 4)):
    """Share of the non-affine overlay energy each basis captures (physics vs generic polynomial).

    All spans are orthonormalized (SVD, relative tolerance 1e-8) so nearly collinear columns cannot
    inflate the fit.
    """
    P = train.P
    A = pb.affine_modes(P)
    Fn = train.F / train.half_L[:, None]
    Qa = pb._orth(A)
    base = Fn - (Fn @ Qa) @ Qa.T
    tot = (base ** 2).sum()
    x, y = P[:, 0], P[:, 1]
    rows = []
    for deg in degs:
        E = pb.eigen_basis(tuple(map(tuple, P)), deg)
        G = pb.gradient_fields(P, deg)
        cols = []
        for k in range(deg + 1):
            for i in range(k + 1):
                m = x ** i * y ** (k - i)
                cols += [np.r_[m, 0 * m], np.r_[0 * m, m]]
        for name, B in [("membrane_eigenstrain", E), ("bending_gradient", G),
                        ("membrane+bending (Kirchhoff-Love)", np.column_stack([E, G])),
                        ("generic_polynomial", np.column_stack(cols))]:
            Q = pb._orth(np.column_stack([A, B]))
            r = Fn - (Fn @ Q) @ Q.T
            rms = np.sqrt((r ** 2).mean(1)) * train.half_L
            rows.append({"basis": name, "degree": deg, "extra_dims": Q.shape[1] - Qa.shape[1],
                         "nonaffine_energy_captured": 1 - (r ** 2).sum() / tot,
                         "median_rms_left_nm": float(np.median(rms)), "p95_rms_left_nm": float(np.percentile(rms, 95))})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out

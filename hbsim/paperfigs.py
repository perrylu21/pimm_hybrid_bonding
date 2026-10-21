"""Paper figures and tables from the benchmark CSVs (results/benchmark). Host only.

Colors follow a fixed categorical order (validated palette); every multi-series figure has a
legend plus direct labels, and the baselines are drawn in neutral gray.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # model -> (label, color, linestyle, marker)
    "physbasis": ("Physics-structured basis + GP (ours)", "#2a78d6", "-", "o"),
    "pinn": ("Parametric PINN (ours)", "#eb6834", "-", "s"),
    "gp_hybrid": ("Analytic + GP residual", "#1baf7a", "--", "^"),
    "mlp_hybrid": ("Analytic + MLP ensemble", "#6250d6", "--", "v"),
    "gp_data_only": ("GP, data only", "#8a8984", ":", "D"),
    "analytic": ("Analytic only (0 FEM runs)", INK2, (0, (1, 3)), None),
    "physbasis_noprior": ("Basis + GP, no analytic prior", "#e87ba4", "-.", "o"),
    "physbasis_membrane_only": ("Basis + GP, membrane only", "#eda100", "-.", "o"),
    "pinn_nophys": ("PINN, no physics loss", "#e34948", "-.", "s"),
    "pinn_nobending": ("PINN, membrane only", "#008300", "-.", "s"),
    "pinn_noprior": ("PINN, no analytic prior", "#9085e9", "-.", "s"),
}
MAIN = ["physbasis", "pinn", "gp_hybrid", "mlp_hybrid", "gp_data_only"]


def _style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                         "lines.linewidth": 2.0, "lines.markersize": 5, "savefig.bbox": "tight"})


def _bench(bench, name):
    import glob
    frames = [pd.read_csv(p) for p in glob.glob(str(Path(bench) / "**" / name), recursive=True)]
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else None


def learning_curves(e1, out):
    panels = [("rmse_m_x", "Magnification error, RMSE [ppm]"), ("rmse_d_max", "d_max error, RMSE [nm]"),
              ("rmse_resid_3sigma", "Non-uniform 3σ error, RMSE [nm]"), ("field_rmse_nm", "Pad-level field RMSE [nm]")]
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, (col, title) in zip(axs, panels):
        for m in MAIN + ["analytic"]:
            d = e1[e1.model == m]
            if d.empty or col not in d or d[col].isna().all():
                continue
            lab, c, ls, mk = STYLE[m]
            if m == "analytic":
                ax.axhline(d[col].median(), color=c, ls=ls, lw=1.2, label=lab)
                continue
            g = d.groupby("n_train")[col]
            x, med, lo, hi = g.median().index, g.median(), g.min(), g.max()
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(x, med, color=c, ls=ls, marker=mk, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FEM training runs N")
        ax.set_title(title, fontsize=9, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    for ax in axs[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_e1_learning_curves.pdf")
    fig.savefig(out / "fig_e1_learning_curves.png", dpi=160)
    plt.close(fig)


def ablations(e1, out):
    groups = [["physbasis", "physbasis_noprior", "physbasis_membrane_only"],
              ["pinn", "pinn_nophys", "pinn_nobending", "pinn_noprior"]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ms in zip(axs, groups):
        for m in ms:
            d = e1[e1.model == m]
            if d.empty:
                continue
            lab, c, ls, mk = STYLE[m]
            g = d.groupby("n_train")["field_rmse_nm"]
            ax.plot(g.median().index, g.median(), color=c, ls=ls, marker=mk, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FEM training runs N")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Pad-level field RMSE [nm]")
    fig.tight_layout()
    fig.savefig(out / "fig_e4_ablations.pdf")
    fig.savefig(out / "fig_e4_ablations.png", dpi=160)
    plt.close(fig)


def representation(r0, out):
    d = r0[r0.degree == r0.degree.max()].copy()
    order = ["membrane_eigenstrain", "bending_gradient", "generic_polynomial", "membrane+bending (Kirchhoff-Love)"]
    names = {"membrane_eigenstrain": "Membrane only", "bending_gradient": "Bending only (curl-free)",
             "generic_polynomial": "Generic polynomial", "membrane+bending (Kirchhoff-Love)": "Membrane + bending (ours)"}
    d = d.set_index("basis").loc[order]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    cols = ["#8a8984", "#8a8984", "#8a8984", "#2a78d6"]
    y = np.arange(len(d))
    ax.barh(y, 100 * d.nonaffine_energy_captured, color=cols, height=0.6)
    for i, (v, k, r) in enumerate(zip(d.nonaffine_energy_captured, d.extra_dims, d.median_rms_left_nm)):
        ax.text(100 * v + 1, i, f"{100 * v:.1f}%  ({k} dims, {r:.2f} nm left)", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [names[b] for b in d.index])
    ax.set_xlim(0, 135)
    ax.set_xlabel("Share of non-affine FEM overlay energy represented [%]")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_r0_representation.pdf")
    fig.savefig(out / "fig_r0_representation.png", dpi=160)
    plt.close(fig)


def sparse_marks(e2, out):
    meth = {"affine": ("Industry 6-parameter overlay model", "#8a8984", "--", "D"),
            "poly2": ("2nd-order correction (12 terms)", "#6250d6", "--", "v"),
            "poly3": ("3rd-order correction (20 terms)", "#e87ba4", "--", "^"),
            "prior_affine": ("Physics prior + affine fit", "#1baf7a", "--", "s"),
            "physics_prior_only": ("Physics prior only (no marks)", INK2, ":", None),
            "physics_map": ("Physics-structured MAP (ours)", "#2a78d6", "-", "o")}
    noises = sorted(e2.noise_nm.unique())
    priors = sorted(e2.n_prior.unique())
    fig, axs = plt.subplots(len(priors), len(noises), figsize=(3.3 * len(noises), 2.8 * len(priors)), squeeze=False,
                            sharey="row")
    for i, npr in enumerate(priors):
        for j, s in enumerate(noises):
            ax = axs[i, j]
            for m, (lab, c, ls, mk) in meth.items():
                d = e2[(e2.method == m) & (e2.noise_nm == s) & (e2.n_prior == npr)].groupby("k").rmse_median_nm.median()
                if not d.empty:
                    ax.plot(d.index, d.values, color=c, ls=ls, marker=mk, label=lab)
            ax.set_title(f"prior from {npr} FEM runs, noise {s:g} nm", fontsize=8.5)
            ax.set_xlabel("overlay marks per die k")
            ax.set_xticks(sorted(e2.k.unique()))
        axs[i, 0].set_ylabel("median field RMSE [nm]")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend below the bottom row's labels
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.savefig(out / "fig_e2_sparse_marks.pdf")
    fig.savefig(out / "fig_e2_sparse_marks.png", dpi=160)
    plt.close(fig)


def robustness(e3, out):
    cases = list(dict.fromkeys(e3.case))
    models = [m for m in MAIN + ["analytic"] if m in set(e3.model)]
    fig, ax = plt.subplots(figsize=(8, 3))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        lab, c, _, _ = STYLE[m]
        v = [e3[(e3.case == cs) & (e3.model == m)].r2_d_max.median() for cs in cases]
        ax.bar(np.arange(len(cases)) + (k - (len(models) - 1) / 2) * w, v, w * 0.92, color=c, label=lab)
    ax.set_xticks(np.arange(len(cases)), [c.replace("_", " ") for c in cases], fontsize=8)
    ax.set_ylabel("holdout R² of d_max")
    ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_e3_robustness.pdf")
    fig.savefig(out / "fig_e3_robustness.png", dpi=160)
    plt.close(fig)


def field_example(train_dir, test_dir, out, n_train=256, rep=0, pct=90):
    """One holdout die (pct-th percentile of non-affine magnitude): FEM vs physics-informed prediction."""
    from hbsim import physbasis as pb
    from hbsim.benchmark import subset_idx
    from hbsim.fielddata import load_fields
    tr, te = load_fields(train_dir), load_fields(test_dir)
    model = pb.PhysicsBasisGP(seed=rep).fit(tr.subset(subset_idx(len(tr), n_train, rep)))
    Fp = model.predict(te.df)
    Qa = pb._orth(pb.affine_modes(te.P))
    na_true = te.F - (te.F @ Qa) @ Qa.T
    mag_all = np.sqrt((na_true ** 2).mean(1))
    t = int(np.argmin(np.abs(mag_all - np.percentile(mag_all, pct))))
    n = len(te.P)
    na_pred = Fp[t] - Qa @ (Qa.T @ Fp[t])
    P = te.P * te.half_L[t]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    vmax = np.abs(na_true[t]).max()
    for ax, f, title in [(axs[0], na_true[t], "FEM (3D, transient)"),
                         (axs[1], na_pred, f"Physics-informed, N = {n_train}"),
                         (axs[2], na_true[t] - na_pred, "Error")]:
        mag = np.sqrt(f[:n] ** 2 + f[n:] ** 2)
        sc = ax.scatter(P[:, 0], P[:, 1], c=mag, s=9, cmap="Blues", vmin=0, vmax=vmax, marker="s")
        ax.quiver(P[::2, 0], P[::2, 1], f[:n][::2], f[n:][::2], color=INK, scale=vmax * 25, width=0.004)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.grid(False)
    axs[0].set_ylabel("y [mm]")
    fig.colorbar(sc, ax=axs, label="non-affine overlay [nm]", shrink=0.85)
    fig.savefig(out / "fig_field_example.pdf")
    fig.savefig(out / "fig_field_example.png", dpi=160)
    plt.close(fig)
    return te.df.iloc[t]


def summary_table(e1, out, sizes=(16, 32, 64, 256)):
    rows = []
    for m in [x for x in MAIN + ["analytic"] if x in set(e1.model)]:
        r = {"model": STYLE[m][0]}
        for n in sizes:
            d = e1[(e1.model == m) & ((e1.n_train == n) | (m == "analytic"))]
            r[f"R2 d_max N={n}"] = d.r2_d_max.median()
            r[f"field RMSE N={n} [nm]"] = d.field_rmse_nm.median() if "field_rmse_nm" in d else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "table_e1_summary.csv", index=False)
    (out / "table_e1_summary.tex").write_text(t.to_latex(index=False, float_format="%.3f", na_rep="--"))
    return t


def make_all(bench, out, train_dir, test_dir):
    _style()
    out.mkdir(parents=True, exist_ok=True)
    r0, e1, e2, e3 = (_bench(bench, f) for f in ("r0_representation.csv", "e1_learning_curve.csv",
                                                  "e2_sparse_marks.csv", "e3_robustness.csv"))
    if r0 is not None:
        representation(r0, out)
    if e1 is not None:
        learning_curves(e1, out)
        ablations(e1, out)
        print(summary_table(e1, out).round(3).to_string(index=False))
    if e2 is not None:
        sparse_marks(e2, out)
    if e3 is not None:
        robustness(e3, out)
    field_example(train_dir, test_dir, out)
    print(f"paper figures written to {out}")

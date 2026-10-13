"""Static paper figures (PDF) from datasets, the surrogate, and Sobol results. Host only.

Colors follow a validated colorblind-safe palette: categorical slots 1-2 (blue, orange),
single-hue blue sequential ramp, recessive gridlines. At most two series per chart, each
identified by legend and label, never by color alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter  # noqa: E402

from hbsim.params import SWEPT, defaults  # noqa: E402
from hbsim.postprocess import OUTPUTS  # noqa: E402
from hbsim.surrogate import INPUTS, ok_rows  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]
TEXT, TEXT_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
UNITS = {"t_x": "nm", "t_y": "nm", "theta": "µrad", "m_x": "ppm", "m_y": "ppm",
         "resid_3sigma": "nm", "d_max": "nm"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight",
})


def log_axis(ax, axis):
    """Plain-number major ticks at 1, 2, 5 x 10^n and no minor labels on a log axis.

    Matplotlib labels minor ticks when a log axis spans under ~2 decades (e.g. t_die 30-775 um),
    and those labels overlap.
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def _label(name):
    unit = UNITS.get(name) or SWEPT[name].unit
    return f"{name} [{unit}]"


def parity(model, df_test, out_path, outputs=("m_x", "d_max", "t_x", "resid_3sigma")):
    """Predicted vs FEM scatter per output with the y = x line; R^2 in each title."""
    df = ok_rows(df_test)
    pred, _ = model.predict(df, with_std=False)
    fig, axes = plt.subplots(1, len(outputs), figsize=(3.2 * len(outputs), 3.2))
    for ax, name in zip(np.atleast_1d(axes), outputs):
        j = OUTPUTS.index(name)
        true = df[name].to_numpy(dtype=float)
        lo, hi = min(true.min(), pred[:, j].min()), max(true.max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=1, label="y = x")
        ax.scatter(true, pred[:, j], s=10, color=SERIES[0], edgecolor="white", linewidth=0.5,
                   zorder=2, label="holdout run")
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = f"{1 - np.sum((pred[:, j] - true) ** 2) / ss_tot:.3f}" if ss_tot > 0 else "n/a (constant)"
        ax.set_title(f"{name}   R² = {r2}", fontsize=9)
        ax.set_xlabel(f"FEM {_label(name)}")
        ax.set_ylabel(f"surrogate {_label(name)}")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def sobol_bars(first_total, output, out_path):
    """Horizontal stacked bars per input: first-order S1 plus interaction share (ST - S1)."""
    d = first_total[first_total["output"] == output].sort_values("ST")
    s1 = d["S1"].clip(lower=0).to_numpy()
    inter = (d["ST"] - d["S1"].clip(lower=0)).clip(lower=0).to_numpy()
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.0, 0.28 * len(d) + 1.0))
    ax.barh(y, s1, height=0.6, color=SERIES[0], edgecolor="white", linewidth=1, label="first-order Sᵢ")
    ax.barh(y, inter, left=s1, height=0.6, color=SERIES[1], edgecolor="white", linewidth=1,
            label="interaction S_Tᵢ − Sᵢ")
    ax.errorbar(d["ST"], y, xerr=d["ST_conf"], fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=2)
    ax.set_yticks(y, d["param"])
    ax.set_xlabel("Sobol index")
    ax.set_title(f"Sensitivity of {_label(output)}", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def s2_heatmap(second_order, output, out_path):
    """Symmetric heatmap of second-order Sobol indices S_ij (negative estimates shown as 0)."""
    d = second_order[second_order["output"] == output]
    n = len(INPUTS)
    M = np.zeros((n, n))
    for _, r in d.iterrows():
        a, b = INPUTS.index(r["param_a"]), INPUTS.index(r["param_b"])
        M[a, b] = M[b, a] = max(r["S2"], 0.0)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, cmap=BLUES, vmin=0)
    ax.set_xticks(range(n), INPUTS, rotation=60, ha="right")
    ax.set_yticks(range(n), INPUTS)
    ax.grid(False)
    top = d.assign(S2=d["S2"].clip(lower=0)).nlargest(5, "S2")  # label only the 5 largest pairs
    for _, r in top.iterrows():
        a, b = INPUTS.index(r["param_a"]), INPUTS.index(r["param_b"])
        for i, k in ((a, b), (b, a)):
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, k] > 0.5 * M.max() else TEXT)
    fig.colorbar(im, ax=ax, label="second-order index S_ij")
    ax.set_title(f"Pairwise interactions for {_label(output)}", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


def process_window(model, x_param, y_param, out_path, output="d_max", spec=100.0, n=81, base=None):
    """Contour map of a surrogate output over two inputs (others at defaults), with a spec line."""
    base = {**defaults(), **(base or {})}
    sx, sy = SWEPT[x_param], SWEPT[y_param]

    def axis(s):
        return np.geomspace(s.low, s.high, n) if s.scale == "log" else np.linspace(s.low, s.high, n)

    X, Y = np.meshgrid(axis(sx), axis(sy))
    df = pd.DataFrame([{**{k: base[k] for k in INPUTS}, x_param: x, y_param: y}
                       for x, y in zip(X.ravel(), Y.ravel())])
    Z = model.predict(df, with_std=False)[0][:, OUTPUTS.index(output)].reshape(X.shape)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    cf = ax.contourf(X, Y, Z, levels=12, cmap=BLUES)
    if Z.min() < spec < Z.max():
        ax.contour(X, Y, Z, levels=[spec], colors=[SERIES[1]], linewidths=2)
        ax.legend(handles=[Line2D([], [], color=SERIES[1], linewidth=2,
                                  label=f"spec {spec:g} {UNITS[output]}")], loc="upper left")
        ax.set_title(f"{output} process window (spec {spec:g} {UNITS[output]})", fontsize=9)
    else:
        side = "everywhere above" if Z.min() >= spec else "everywhere below"
        ax.set_title(f"{output} {side} spec {spec:g} {UNITS[output]}", fontsize=9)
    if sx.scale == "log":
        ax.set_xscale("log")
        log_axis(ax, "x")
    if sy.scale == "log":
        ax.set_yscale("log")
        log_axis(ax, "y")
    ax.set_xlabel(_label(x_param))
    ax.set_ylabel(_label(y_param))
    ax.grid(False)
    fig.colorbar(cf, ax=ax, label=f"surrogate {_label(output)}")
    fig.savefig(out_path)
    plt.close(fig)


def line_check(model, df_line, param, out_path, output="d_max"):
    """FEM one-at-a-time sweep (points) against the surrogate mean and ±2σ ensemble band."""
    df = ok_rows(df_line).sort_values(param)
    mean, std = model.predict(df)
    j = OUTPUTS.index(output)
    x = df[param].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.fill_between(x, mean[:, j] - 2 * std[:, j], mean[:, j] + 2 * std[:, j], color=SERIES[0],
                    alpha=0.2, linewidth=0, label="surrogate ±2σ")
    ax.plot(x, mean[:, j], color=SERIES[0], label="surrogate mean")
    ax.scatter(x, df[output], s=24, color=SERIES[1], edgecolor="white", linewidth=1, zorder=3,
               label="FEM")
    if SWEPT[param].scale == "log":
        ax.set_xscale("log")
        log_axis(ax, "x")
    ax.set_xlabel(_label(param))
    ax.set_ylabel(_label(output))
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def learning_curve_plot(curve, out_path, outputs=("m_x", "d_max")):
    """Holdout RMSE vs number of training runs (log-log), one line per output."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for color, name in zip(SERIES, outputs):
        d = curve[curve["output"] == name]
        ax.plot(d["n_train"], d["rmse"], color=color, marker="o", markersize=5, label=_label(name))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("training runs")
    ax.set_ylabel("holdout RMSE")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

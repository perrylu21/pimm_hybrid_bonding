#!/usr/bin/env python3
"""Command-line entry point for the hybrid-bonding die-shift study.

  python cli.py run studies/pilot.yaml [--retry-failed]
  python cli.py collect results/pilot
  python cli.py train --train results/main_sweep --test results/holdout
  python cli.py sensitivity --surrogate results/main_sweep/surrogate
  python cli.py figures --surrogate results/main_sweep/surrogate --test results/holdout
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _dataset(study_dir):
    return pd.read_parquet(Path(study_dir) / "dataset.parquet")


def cmd_run(a):
    from hbsim import params, runner
    runner.run_study(params.load_study(a.study), retry_failed=a.retry_failed)


def cmd_collect(a):
    from hbsim import runner
    df = runner.collect(a.study_dir)
    print(df["status"].value_counts().to_string())


def cmd_train(a):
    from hbsim import surrogate
    out = Path(a.out or Path(a.train) / "surrogate")
    model = surrogate.Surrogate.fit(_dataset(a.train), n_models=a.n_models)
    model.save(out)
    report = {"surrogate": surrogate.evaluate(model, _dataset(a.test))}
    if a.gp:
        gp = surrogate.GPBaseline.fit(_dataset(a.train))
        report["gp"] = surrogate.evaluate(gp, _dataset(a.test))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    if a.learning_curve:
        surrogate.learning_curve(_dataset(a.train), _dataset(a.test), n_models=a.n_models).to_csv(
            out / "learning_curve.csv", index=False)
    for name, s in report["surrogate"].items():
        print(f"{name:14s} R2={s['r2']:.4f}  RMSE={s['rmse']:.4g}")


def cmd_sensitivity(a):
    from hbsim import sensitivity, surrogate
    model = surrogate.Surrogate.load(a.surrogate)
    ft, so = sensitivity.sobol_indices(lambda df: model.predict(df)[0], n_base=a.n_base, seed=a.seed)
    ft.to_csv(Path(a.surrogate) / "sobol_first_total.csv", index=False)
    so.to_csv(Path(a.surrogate) / "sobol_second_order.csv", index=False)
    for name in ("m_x", "d_max"):
        top = ft[ft["output"] == name].sort_values("ST", ascending=False).head(5)
        print(f"\n{name}:\n" + top[["param", "S1", "ST", "interaction"]].round(3).to_string(index=False))


def cmd_figures(a):
    from hbsim import figures, surrogate
    sdir = Path(a.surrogate)
    out = figures.ensure_dir(a.out or sdir.parent / "figures")
    model = surrogate.Surrogate.load(sdir)
    figures.parity(model, _dataset(a.test), out / "parity.pdf")
    ft_path, so_path = sdir / "sobol_first_total.csv", sdir / "sobol_second_order.csv"
    if ft_path.exists():
        ft, so = pd.read_csv(ft_path), pd.read_csv(so_path)
        for name in ("m_x", "d_max"):
            figures.sobol_bars(ft, name, out / f"sobol_{name}.pdf")
            figures.s2_heatmap(so, name, out / f"s2_{name}.pdf")
    if (sdir / "learning_curve.csv").exists():
        figures.learning_curve_plot(pd.read_csv(sdir / "learning_curve.csv"), out / "learning_curve.pdf")
    figures.process_window(model, "T_head", "bow_h", out / "window_Thead_bow.pdf", spec=a.spec)
    figures.process_window(model, "bow_h", "k_head", out / "window_bow_khead.pdf", spec=a.spec)
    for line_dir in a.lines:
        df = _dataset(line_dir)
        param = json.loads((Path(line_dir) / "manifest.json").read_text())["study"]["sampler"]["param"]
        figures.line_check(model, df, param, out / f"line_{param}.pdf")
    print(f"figures written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("run", help="execute a study YAML")
    p.add_argument("study")
    p.add_argument("--retry-failed", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect", help="rebuild dataset.parquet for a study directory")
    p.add_argument("study_dir")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("train", help="train the surrogate and score it on a holdout study")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--gp", action="store_true", help="also fit and score the GP baseline")
    p.add_argument("--learning-curve", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("sensitivity", help="Sobol indices on a trained surrogate")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--n-base", type=int, default=2 ** 14)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_sensitivity)
    p = sub.add_parser("figures", help="write paper figures")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--lines", nargs="*", default=[])
    p.add_argument("--spec", type=float, default=100.0, help="d_max spec line [nm]")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_figures)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

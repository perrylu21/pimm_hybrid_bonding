#!/usr/bin/env python3
"""Generate docs/paper/numbers.tex from results/benchmark/*.csv.

Every number quoted in the paper is a LaTeX macro defined here, so rerunning the benchmarks and
this script updates the paper. Medians are over repetitions.

    .venv/bin/python scripts/make_numbers.py [--bench results/benchmark] [--out docs/paper/numbers.tex]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

BASELINES = ("gp_data_only", "gp_hybrid", "mlp_hybrid")  # learned baselines (analytic alone excluded)
OURS = "physbasis"


def first_n_reaching(df, model, col, threshold):
    """Smallest n_train whose median `col` over reps is >= threshold, or None."""
    med = df[df.model == model].groupby("n_train")[col].median().sort_index()
    hit = med[med >= threshold]
    return int(hit.index[0]) if len(hit) else None


def _median(df, col, **where):
    sel = df
    for k, v in where.items():
        sel = sel[sel[k] == v]
    if sel.empty:
        raise ValueError(f"no rows for {where}")
    return float(sel[col].median())


def compute(bench):
    bench = Path(bench)
    n = {}

    r0 = pd.read_csv(bench / "r0_representation.csv")

    def r0v(basis, degree, col):
        row = r0[(r0.basis == basis) & (r0.degree == degree)]
        return float(row[col].iloc[0])

    kl = "membrane+bending (Kirchhoff-Love)"
    n["RzeroKL"] = (100 * r0v(kl, 4, "nonaffine_energy_captured"), "{:.1f}", "R0 membrane+bending, degree 4: non-affine energy captured [%]")
    n["RzeroKLDims"] = (int(r0v(kl, 4, "extra_dims")), "{:d}", "R0 membrane+bending degree 4: dimensions")
    n["RzeroGeneric"] = (100 * r0v("generic_polynomial", 4, "nonaffine_energy_captured"), "{:.1f}", "R0 generic polynomial, degree 4 [%]")
    n["RzeroGenericDims"] = (int(r0v("generic_polynomial", 4, "extra_dims")), "{:d}", "R0 generic polynomial degree 4: dimensions")
    n["RzeroMembrane"] = (100 * r0v("membrane_eigenstrain", 4, "nonaffine_energy_captured"), "{:.1f}", "R0 membrane only, degree 4 [%]")
    n["RzeroKLResidNm"] = (r0v(kl, 4, "median_rms_left_nm"), "{:.2f}", "R0 membrane+bending: median RMS left [nm]")
    n["RzeroGenericResidNm"] = (r0v("generic_polynomial", 4, "median_rms_left_nm"), "{:.2f}", "R0 generic polynomial: median RMS left [nm]")

    e1 = pd.read_csv(bench / "e1_learning_curve.csv")
    n["NdmaxOurs"] = (first_n_reaching(e1, OURS, "r2_d_max", 0.95), "{:d}", "E1: FEM runs to reach median R2(d_max) >= 0.95, ours")
    base_ns = [first_n_reaching(e1, m, "r2_d_max", 0.95) for m in BASELINES]
    base_ns = [v for v in base_ns if v is not None]
    n["NdmaxBaseline"] = (min(base_ns) if base_ns else None, "{:d}", "E1: same, best learned baseline")
    n["EoneOursThirtyTwo"] = (_median(e1, "r2_d_max", model=OURS, n_train=32), "{:.3f}", "E1: R2(d_max) at N=32, ours")
    n["EoneBaseThirtyTwo"] = (max(_median(e1, "r2_d_max", model=m, n_train=32) for m in BASELINES), "{:.3f}", "E1: R2(d_max) at N=32, best learned baseline")
    n["EoneMembraneDmaxSixtyFour"] = (_median(e1, "r2_d_max", model="physbasis_membrane_only", n_train=64), "{:.3f}", "E1: R2(d_max) at N=64, membrane-only basis")
    n["EoneOursDmaxSixtyFour"] = (_median(e1, "r2_d_max", model=OURS, n_train=64), "{:.3f}", "E1: R2(d_max) at N=64, ours")
    n["EoneNoPriorDmaxSixtyFour"] = (_median(e1, "r2_d_max", model="physbasis_noprior", n_train=64), "{:.3f}", "E1: R2(d_max) at N=64, ours without analytic prior")
    n["EoneOursResidOneTwoEight"] = (_median(e1, "r2_resid_3sigma", model=OURS, n_train=128), "{:.3f}", "E1: R2(resid_3sigma) at N=128, ours")
    n["EoneMembraneResidOneTwoEight"] = (_median(e1, "r2_resid_3sigma", model="physbasis_membrane_only", n_train=128), "{:.3f}", "E1: R2(resid_3sigma) at N=128, membrane-only")
    n["EoneOursFieldFiveTwelve"] = (_median(e1, "field_rmse_nm", model=OURS, n_train=512), "{:.1f}", "E1: field RMSE at N=512, ours [nm]")
    n["EoneMembraneFieldFiveTwelve"] = (_median(e1, "field_rmse_nm", model="physbasis_membrane_only", n_train=512), "{:.1f}", "E1: field RMSE at N=512, membrane-only [nm]")

    e2 = pd.read_csv(bench / "e2_sparse_marks.csv")

    def e2v(prior, k, noise, method):
        sel = e2[(e2.n_prior == prior) & (e2.k == k) & (e2.noise_nm == noise) & (e2.method == method)]
        if sel.empty:
            raise ValueError(f"no E2 rows for prior={prior} k={k} noise={noise} {method}")
        return float(sel.rmse_median_nm.mean())

    n["EtwoOursFour"] = (e2v(256, 4, 2.0, "physics_map"), "{:.1f}", "E2: 4 marks, 2 nm noise, 256-run prior: median RMSE [nm]")
    n["EtwoOursFourSmallPrior"] = (e2v(64, 4, 2.0, "physics_map"), "{:.1f}", "E2: same with 64-run prior [nm]")
    n["EtwoAffineFour"] = (e2v(256, 4, 2.0, "affine"), "{:.1f}", "E2: 4 marks, 2 nm noise, affine overlay model [nm]")
    n["EtwoPriorOnly"] = (e2v(256, 4, 2.0, "physics_prior_only"), "{:.1f}", "E2: 256-run prior, no marks [nm]")
    for prior, ptag in ((256, ""), (64, "SmallPrior")):
        for k, ktag in ((4, "Four"), (9, "Nine"), (16, "Sixteen")):
            if (k, ptag) not in ((4, ""), (4, "SmallPrior")):  # defined above
                n[f"EtwoOurs{ktag}{ptag}"] = (e2v(prior, k, 2.0, "physics_map"), "{:.1f}",
                                               f"E2: {k} marks, 2 nm noise, {prior}-run prior, ours [nm]")
            n[f"EtwoPriorAffine{ktag}{ptag}"] = (e2v(prior, k, 2.0, "prior_affine"), "{:.1f}",
                                                  f"E2: {k} marks, 2 nm noise, {prior}-run prior + affine fit [nm]")
    n["EtwoPolyTwoNine"] = (e2v(256, 9, 2.0, "poly2"), "{:.1f}", "E2: 9 marks, 2 nm noise, 2nd-order correction [nm]")
    n["EtwoPolyThreeSixteen"] = (e2v(256, 16, 2.0, "poly3"), "{:.1f}", "E2: 16 marks, 2 nm noise, 3rd-order correction [nm]")
    n["EtwoAffineSixteen"] = (e2v(256, 16, 2.0, "affine"), "{:.1f}", "E2: 16 marks, 2 nm noise, affine [nm]")

    e3 = pd.read_csv(bench / "e3_robustness.csv")
    for case, tag in (("ood_dT_head_gt40", "OodDT"), ("ood_L_die_gt14", "OodL"), ("train_noise_5nm", "NoiseFive")):
        n[f"Ethree{tag}Ours"] = (_median(e3, "r2_d_max", case=case, model=OURS), "{:.2f}", f"E3 {case}: R2(d_max), ours")
        vals = [_median(e3, "r2_d_max", case=case, model=m) for m in BASELINES]
        n[f"Ethree{tag}BaseMin"] = (min(vals), "{:.2f}", f"E3 {case}: R2(d_max), worst learned baseline")
        n[f"Ethree{tag}BaseMax"] = (max(vals), "{:.2f}", f"E3 {case}: R2(d_max), best learned baseline")

    pinn = pd.read_csv(bench / "pinn" / "e1_learning_curve.csv")
    abl = pd.read_csv(bench / "pinn_ablations" / "e1_learning_curve.csv")
    n["PinnFieldPhys"] = (_median(pinn, "field_rmse_nm", model="pinn", n_train=64), "{:.0f}", "E4: PINN field RMSE at N=64, with physics loss [nm]")
    n["PinnFieldNoPhys"] = (_median(abl, "field_rmse_nm", model="pinn_nophys", n_train=64), "{:.0f}", "E4: PINN field RMSE at N=64, physics loss removed [nm]")
    n["PinnDmaxSixtyFour"] = (_median(pinn, "r2_d_max", model="pinn", n_train=64), "{:.2f}", "E4: PINN R2(d_max) at N=64")
    return n


def render(numbers):
    lines = ["% Generated by scripts/make_numbers.py from results/benchmark/*.csv -- do not edit by hand."]
    for name, (value, fmt, comment) in numbers.items():
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} must contain letters only (LaTeX)")
        if value is None:
            raise ValueError(f"{name}: value missing (threshold never reached or data absent)")
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value)}}}  % {comment}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="results/benchmark")
    ap.add_argument("--out", default="docs/paper/numbers.tex")
    a = ap.parse_args()
    text = render(compute(a.bench))
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

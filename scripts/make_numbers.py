#!/usr/bin/env python3
"""Generate docs/paper/numbers.tex from results/benchmark/*.csv.

Every number quoted in the paper is a LaTeX macro defined here, so rerunning the benchmarks and
this script updates the paper. Medians are over repetitions.

    .venv/bin/python scripts/make_numbers.py [--bench results/benchmark] [--out docs/paper/numbers.tex]
"""
import argparse
import re
import sys
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
    for k, word in ((4, "Four"), (9, "Nine"), (16, "Sixteen")):
        if k not in set(e2.k):
            raise ValueError(f"E2 has no k={k} rows")
        n[f"EtwoK{word}"] = (k, "{:d}", f"E2 mark count quoted in the text")
    n["EtwoNoiseHigh"] = (float(e2.noise_nm.max()), "{:g}", "E2 highest noise level [nm]")
    n["EtwoNoiseRef"] = (2.0, "{:g}", "E2 noise level used for quoted numbers [nm]") if 2.0 in set(e2.noise_nm) else (None, "{:g}", "")
    n["EtwoOursNineNoiseHigh"] = (e2v(256, 9, float(e2.noise_nm.max()), "physics_map"), "{:.1f}", "E2: 9 marks, highest noise, 256-run prior, ours [nm]")
    n["EtwoPriorAffineNineNoiseHigh"] = (e2v(256, 9, float(e2.noise_nm.max()), "prior_affine"), "{:.1f}", "E2: same, prior + affine [nm]")
    # Ratios for the two-part E2 message (user framing (a), 2026-11-12), from unrounded means.
    for prior, ptag in ((256, "Large"), (64, "Small")):
        n[f"EtwoPriorGainFour{ptag}"] = (e2v(prior, 4, 2.0, "affine") / e2v(prior, 4, 2.0, "prior_affine"), "{:.1f}",
                                          f"E2: 4 marks, affine / (prior + affine), {prior}-run prior")
        n[f"EtwoBasisGainSixteen{ptag}"] = (e2v(prior, 16, 2.0, "prior_affine") / e2v(prior, 16, 2.0, "physics_map"), "{:.1f}",
                                             f"E2: 16 marks, (prior + affine) / ours, {prior}-run prior")
        n[f"EtwoBasisGainNine{ptag}"] = (e2v(prior, 9, 2.0, "prior_affine") / e2v(prior, 9, 2.0, "physics_map"), "{:.2f}",
                                          f"E2: 9 marks, (prior + affine) / ours, {prior}-run prior")

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

    # Experiment configuration, read back from the result files (what was actually run).
    def lst(vals):
        return ", ".join(f"{v:g}" for v in sorted(set(vals)))

    def reps(df, **where):
        sel = df
        for k, v in where.items():
            sel = sel[sel[k] == v]
        return int(sel.rep.nunique())

    n["EoneSizes"] = (lst(e1.n_train), "{}", "E1 training-set sizes")
    n["EoneRepsSmall"] = (reps(e1, model=OURS, n_train=int(e1.n_train.min())), "{:d}", "E1 repeats at small N")
    full = [k for k in sorted(e1[e1.model == OURS].n_train.unique()) if reps(e1, model=OURS, n_train=k) == n["EoneRepsSmall"][0]]
    n["EoneFullRepsMaxN"] = (int(max(full)), "{:d}", "largest E1 size with the full number of repeats")
    sizes = sorted(e1[e1.model == OURS].n_train.unique())
    mid = [k for k in sizes if max(full) < k < max(sizes)]
    if len(mid) != 1:
        raise ValueError(f"expected one E1 size between full-repeat sizes and the largest, got {mid}")
    n["EoneMidN"] = (int(mid[0]), "{:d}", "E1 size with reduced repeats")
    n["EoneRepsFiveTwelve"] = (reps(e1, model=OURS, n_train=int(mid[0])), "{:d}", "E1 repeats at that size")
    n["EoneRepsMax"] = (reps(e1, model=OURS, n_train=int(e1.n_train.max())), "{:d}", "E1 repeats at the largest N")
    n["EoneMaxN"] = (int(e1.n_train.max()), "{:d}", "largest E1 training set")
    n["EtwoKs"] = (lst(e2.k), "{}", "E2 marks per die")
    n["EtwoNoises"] = (lst(e2.noise_nm), "{}", "E2 measurement noise levels [nm]")
    n["EtwoPriorSmall"] = (int(e2.n_prior.min()), "{:d}", "E2 smaller prior training set")
    n["EtwoPriorLarge"] = (int(e2.n_prior.max()), "{:d}", "E2 larger prior training set")
    n["EtwoReps"] = (int(e2.rep.nunique()), "{:d}", "E2 prior subsets per prior size")
    ood = {c: float(re.search(r"gt(\d+(?:\.\d+)?)$", c).group(1)) for c in e3["case"].unique() if c.startswith("ood_")}
    noisec = sorted(float(re.search(r"noise_([\d.]+)nm", c).group(1)) for c in e3["case"].unique() if "noise" in c)
    n["EthreeN"] = (int(e3.n_train.max()), "{:d}", "E3 training-set size")
    n["EthreeReps"] = (int(e3.rep.nunique()), "{:d}", "E3 repeats")
    n["EthreeNoises"] = (" and ".join(f"{v:g}" for v in noisec), "{}", "E3 training-noise levels [nm]")
    n["EthreeDTThreshold"] = (ood["ood_dT_head_gt40"], "{:g}", "E3 extrapolation split on dT_head [C]")
    n["EthreeLThreshold"] = (ood["ood_L_die_gt14"], "{:g}", "E3 extrapolation split on L_die [mm]")
    for case, tag in (("ood_dT_head_gt40", "OodDT"), ("ood_L_die_gt14", "OodL")):
        n[f"Ethree{tag}NTest"] = (int(e3[e3.case == case].n_test.iloc[0]), "{:d}", f"E3 {case}: holdout runs beyond the split")
    n["PinnSizes"] = (lst(pinn.n_train), "{}", "PINN training-set sizes")
    n["PinnReps"] = (int(pinn.rep.nunique()), "{:d}", "PINN repeats per size")
    n["PinnAblationN"] = (int(abl.n_train.iloc[0]), "{:d}", "PINN ablation training-set size")
    n["PinnAblationReps"] = (int(abl.rep.nunique()), "{:d}", "PINN ablation repeats")
    return n


def coarse_vs_finest_pct(conv, coarse):
    """Worst relative difference [%] of m_x and d_max between mesh_n=coarse and the finest mesh, over cases."""
    mesh = conv[conv.setting == "mesh_n"].copy()
    mesh["level"] = mesh.level.astype(int)
    worst = 0.0
    for _, g in mesh.groupby("case"):
        fine = g[g.level == g.level.max()].iloc[0]
        c = g[g.level == coarse].iloc[0]
        for col in ("m_x", "d_max"):
            worst = max(worst, abs(c[col] - fine[col]) / abs(fine[col]) * 100)
    return worst


def compute_reference(root):
    """Numbers about the reference FEM and its datasets (results/convergence.csv, study datasets)."""
    root = Path(root)
    sys.path.insert(0, str(root))
    from hbsim.params import SWEPT  # noqa: E402  (schema is the source of the parameter count)
    n = {}
    train = pd.read_parquet(root / "results" / "main_sweep_fast" / "dataset.parquet")
    hold = pd.read_parquet(root / "results" / "holdout_fast" / "dataset.parquet")
    train, hold = train[train.status == "ok"], hold[hold.status == "ok"]
    coarse = int(train.mesh_n.iloc[0])
    n["NParams"] = (len(SWEPT), "{:d}", "swept process/geometry/material parameters")
    n["NTrainRuns"] = (len(train), "{:d}", "FEM training runs (Sobol)")
    n["NHoldoutRuns"] = (len(hold), "{:d}", "FEM holdout runs (Latin hypercube)")
    n["NPads"] = (int(train.n_pads.iloc[0]), "{:d}", "pads per die edge (n x n grid)")
    n["MeshCoarse"] = (coarse, "{:d}", "elements along the die edge in the datasets")
    n["RunTimeMedianS"] = (float(train.runtime_s.median()), "{:.0f}", "median wall time per FEM run [s]")
    conv = pd.read_csv(root / "results" / "convergence.csv")
    mesh = conv[conv.setting == "mesh_n"]
    n["MeshFine"] = (int(mesh.level.astype(int).max()), "{:d}", "finest mesh in the convergence study")
    n["ConvCoarsePct"] = (coarse_vs_finest_pct(conv, coarse), "{:.1f}", "worst m_x/d_max change, coarse vs finest mesh [%]")
    d = conv[(conv.case == "default") & (conv.setting == "mesh_n")].copy()
    d["level"] = d.level.astype(int)
    n["DofsCoarse"] = (int(d[d.level == coarse].n_dofs.iloc[0]), "{:,}", "displacement DOFs, die + wafer, coarse mesh")
    si = conv[conv.setting == "bond_layer_as_si"].set_index("case")
    ref = mesh[mesh.level.astype(int) == 20].set_index("case")
    n["BondLayerEffectPct"] = (float((abs(si.d_max - ref.d_max) / ref.d_max * 100).max()), "{:.1f}",
                               "largest d_max change when the bond layers get Si properties [%]")
    # Model constants, read from the code that uses them (never typed into the paper).
    from hbsim.materials import CU, SI, T_REF
    from hbsim.params import NUMERICS
    t_first = float(re.search(r"^T_FIRST = ([0-9.eE+-]+)", (root / "hbsim" / "physics.py").read_text(), re.M).group(1))
    n["SiE"] = (SI["E"] / 1e3, "{:.0f}", "Si Young's modulus [GPa]")
    n["SiNu"] = (SI["nu"], "{:.2f}", "Si Poisson ratio")
    n["SiAlpha"] = (SI["alpha"] * 1e6, "{:.1f}", "Si thermal expansion [ppm/K]")
    n["CuE"] = (CU["E"] / 1e3, "{:.0f}", "Cu Young's modulus [GPa]")
    n["CuAlpha"] = (CU["alpha"] * 1e6, "{:.1f}", "Cu thermal expansion [ppm/K]")
    n["Tref"] = (T_REF, "{:.0f}", "reference (pad patterning) temperature [C]")
    n["WaferPatchFactor"] = (1 + 2 * NUMERICS["margin_frac"], "{:g}", "wafer patch edge / die edge")
    n["NzDie"] = (NUMERICS["nz_die"], "{:d}", "elements through the die thickness")
    n["NzWafer"] = (NUMERICS["nz_wafer"], "{:d}", "elements through the wafer thickness")
    n["NRing"] = (NUMERICS["n_ring"], "{:d}", "elements across the wafer margin")
    n["NTimeSteps"] = (int(train.n_t.iloc[0]), "{:d}", "thermal time steps per run")
    n["TFirstUs"] = (t_first * 1e6, "{:g}", "first thermal time step [us]")
    n["VerifTolPct"] = (1, "{:d}", "tolerance of the analytic checks in tests/test_verification.py [%]")
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
    text = render({**compute(a.bench), **compute_reference(Path(__file__).resolve().parent.parent)})
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

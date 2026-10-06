"""Global sensitivity analysis (Sobol indices) on the surrogate with SALib. Host only."""
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from hbsim.params import from_unit
from hbsim.postprocess import OUTPUTS
from hbsim.surrogate import INPUTS


def problem():
    return {"num_vars": len(INPUTS), "names": INPUTS, "bounds": [[0.0, 1.0]] * len(INPUTS)}


def decode(U):
    """Unit-cube samples (n, D) -> DataFrame of parameters in user units."""
    return pd.DataFrame({n: [from_unit(n, u) for u in U[:, j]] for j, n in enumerate(INPUTS)})


def sobol_indices(predict, n_base=2 ** 14, seed=0, outputs=OUTPUTS):
    """Sobol S1, ST, interaction (ST - S1) and second-order S2 for each output.

    predict: function(DataFrame) -> (n, len(OUTPUTS)) array, e.g. lambda df: model.predict(df)[0].
    Returns (first_total, second_order) DataFrames.
    """
    prob = problem()
    U = sobol_sample.sample(prob, n_base, calc_second_order=True, seed=seed)
    Y = predict(decode(U))
    ft, so = [], []
    for name in outputs:
        j = OUTPUTS.index(name)
        res = sobol_analyze.analyze(prob, Y[:, j], calc_second_order=True, seed=seed)
        for i, p in enumerate(INPUTS):
            ft.append({"output": name, "param": p, "S1": res["S1"][i], "S1_conf": res["S1_conf"][i],
                       "ST": res["ST"][i], "ST_conf": res["ST_conf"][i],
                       "interaction": res["ST"][i] - res["S1"][i]})
        for a in range(len(INPUTS)):
            for b in range(a + 1, len(INPUTS)):
                so.append({"output": name, "param_a": INPUTS[a], "param_b": INPUTS[b],
                           "S2": res["S2"][a, b], "S2_conf": res["S2_conf"][a, b]})
    return pd.DataFrame(ft), pd.DataFrame(so)

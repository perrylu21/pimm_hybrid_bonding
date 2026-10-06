import math

import pytest

pytest.importorskip("scipy")
from hbsim import params as P  # noqa: E402
from hbsim import sampling  # noqa: E402


def _study(sampler, params=None):
    return P.study_from_dict({"name": "s", "sampler": sampler, "params": params or {}})


def test_sobol_reproducible_in_range_and_resolved():
    st = _study({"type": "sobol", "n": 16, "seed": 42}, {"t_bond": {"value": 0.2}})
    a, b = sampling.generate(st), sampling.generate(st)
    assert a == b and len(a) == 16
    for p in a:
        P.validate(p)
        assert p["t_bond"] == 0.2 and p["mesh_n"] == P.NUMERICS["mesh_n"]


def test_sobol_requires_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        sampling.generate(_study({"type": "sobol", "n": 10, "seed": 1}))


def test_lhs_seed_changes_samples():
    a = sampling.generate(_study({"type": "lhs", "n": 8, "seed": 1}))
    b = sampling.generate(_study({"type": "lhs", "n": 8, "seed": 2}))
    assert a != b


def test_line_varies_one_param_over_range():
    rows = sampling.generate(_study({"type": "line", "param": "v_wave", "n": 5}))
    vals = [r["v_wave"] for r in rows]
    assert math.isclose(vals[0], 5.0) and math.isclose(vals[-1], 200.0)
    assert math.isclose(vals[2], math.sqrt(5.0 * 200.0))  # log-scale midpoint
    assert all(r["T_head"] == P.SWEPT["T_head"].default for r in rows)

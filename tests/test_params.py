import math

import pytest

from hbsim import params as P


def test_unit_round_trip_linear_and_log():
    for name in ("dT_head", "kappa", "v_wave", "t_bond", "k_head"):
        s = P.SWEPT[name]
        for v in (s.low, s.default, s.high):
            assert math.isclose(P.from_unit(name, P.to_unit(name, v)), v, rel_tol=1e-12)


def test_log_scale_midpoint_is_geometric_mean():
    s = P.SWEPT["k_head"]
    assert math.isclose(P.from_unit("k_head", 0.5), math.sqrt(s.low * s.high), rel_tol=1e-12)


def test_defaults_validate():
    P.validate(P.defaults())


@pytest.mark.parametrize("change, message", [
    ({"dT_head": 150.0}, "outside"),
    ({"t_die": 30.0, "t_bond": 5.0}, "outside"),
    ({"kappa": float("nan")}, "finite"),
    ({"mesh_n": 21}, "even"),
    ({"n_t": 0}, "positive"),
])
def test_validate_rejects_bad_values(change, message):
    with pytest.raises(ValueError, match=message):
        P.validate({**P.defaults(), **change})


def test_validate_rejects_missing_param():
    p = P.defaults()
    del p["k_head"]
    with pytest.raises(ValueError, match="missing"):
        P.validate(p)


def test_run_id_stable_and_order_independent():
    p = P.defaults()
    q = dict(reversed(list(p.items())))
    assert P.run_id(p) == P.run_id(q)
    assert len(P.run_id(p)) == 12
    assert P.run_id({**p, "dT_head": 21.0}) != P.run_id(p)


def test_study_pins_narrows_and_rejects_unknown():
    st = P.study_from_dict({"name": "s", "sampler": {"type": "lhs", "n": 4, "seed": 1},
                            "params": {"t_bond": {"value": 0.2}, "dT_head": {"range": [10, 30]}}})
    assert st.pinned == {"t_bond": 0.2}
    assert "t_bond" not in st.specs
    assert (st.specs["dT_head"].low, st.specs["dT_head"].high) == (10.0, 30.0)
    with pytest.raises(ValueError, match="unknown params"):
        P.study_from_dict({"name": "s", "sampler": {}, "params": {"T_hed": {"value": 1}}})
    with pytest.raises(ValueError, match="within schema"):
        P.study_from_dict({"name": "s", "sampler": {}, "params": {"dT_head": {"range": [-10, 30]}}})
    with pytest.raises(ValueError, match="missing required key"):
        P.study_from_dict({"sampler": {}})


def test_schema_uses_temperature_difference_and_curvature():
    assert "T_head" not in P.SWEPT and "bow_h" not in P.SWEPT
    dT, chuck, kappa = P.SWEPT["dT_head"], P.SWEPT["T_chuck"], P.SWEPT["kappa"]
    assert (dT.low, dT.high, dT.unit) == (0.0, 60.0, "degC")
    assert (chuck.low, chuck.high) == (20.0, 60.0)
    assert (kappa.low, kappa.high, kappa.unit) == (-0.5, 0.5, "1/m")
    assert len(P.SWEPT) == 16

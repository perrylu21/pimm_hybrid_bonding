import numpy as np

from hbsim import postprocess as pp


def test_pad_grid_shape_and_inset():
    xy = pp.pad_grid(10.0, 21)
    assert xy.shape == (441, 2)
    assert np.isclose(xy.max(), 4.9) and np.isclose(xy.min(), -4.9)


def test_fit_recovers_synthetic_overlay_exactly():
    xy = pp.pad_grid(10.0, 11)
    tx, ty, th, mx, my = 3e-6, -2e-6, 4e-7, 5e-5, -3e-5
    d = np.column_stack([tx - th * xy[:, 1] + mx * xy[:, 0], ty + th * xy[:, 0] + my * xy[:, 1]])
    f = pp.fit_overlay(xy, d)
    for key, val in zip(("t_x", "t_y", "theta", "m_x", "m_y"), (tx, ty, th, mx, my)):
        assert np.isclose(f[key], val, rtol=1e-9, atol=1e-15)
    assert np.abs(f["resid"]).max() < 1e-15


def test_metrics_units():
    xy = pp.pad_grid(10.0, 11)
    d = np.column_stack([1e-6 + 2e-5 * xy[:, 0], 2e-5 * xy[:, 1]])  # 1 nm shift, 20 ppm
    m = pp.metrics(xy, d)
    assert np.isclose(m["t_x"], 1.0) and np.isclose(m["m_x"], 20.0) and np.isclose(m["m_y"], 20.0)
    assert m["resid_3sigma"] < 1e-6
    assert all(isinstance(v, float) for v in m.values())


def test_remove_rigid_keeps_magnification():
    xy = pp.pad_grid(10.0, 11)
    d = np.column_stack([5e-6 - 1e-6 * xy[:, 1] + 1e-5 * xy[:, 0], 1e-6 * xy[:, 0]])
    out = pp.remove_rigid(xy, d, pp.fit_overlay(xy, d))
    f = pp.fit_overlay(xy, out)
    assert abs(f["t_x"]) < 1e-15 and abs(f["theta"]) < 1e-15 and np.isclose(f["m_x"], 1e-5)


def test_apply_placement_adds_rigid_terms():
    xy = pp.pad_grid(10.0, 5)
    out = pp.apply_placement(xy, np.zeros_like(xy), 10.0, -5.0, 2.0)
    f = pp.fit_overlay(xy, out)
    assert np.isclose(f["t_x"] * 1e6, 10.0) and np.isclose(f["t_y"] * 1e6, -5.0)
    assert np.isclose(f["theta"] * 1e6, 2.0)

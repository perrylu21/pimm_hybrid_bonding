"""Physics-informed field models: basis properties, analytic limits, short PINN training."""
import numpy as np
import pandas as pd
import pytest
import torch

from hbsim import physbasis as pb
from hbsim.fielddata import INPUTS, FieldSet, ref_pads
from hbsim.params import defaults

P = ref_pads(11)


def _df(n, rng):
    base = {k: v for k, v in defaults().items() if k in INPUTS}
    rows = []
    for _ in range(n):
        r = dict(base)
        r["dT_head"] = rng.uniform(0, 60)
        r["L_die"] = rng.uniform(2, 20)
        rows.append(r)
    return pd.DataFrame(rows)


def test_uniform_eigenstrain_column_is_pure_magnification():
    B = pb.full_basis(P)
    assert np.allclose(B[:, 0], np.r_[P[:, 0], P[:, 1]])


def test_basis_is_well_conditioned():
    # near-collinear columns once made a least-squares fit "explain" noise; guard against it
    assert np.linalg.cond(pb.full_basis(P)) < 1e3


def test_gradient_fields_are_curl_free():
    x = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x, x, indexing="ij")
    G = pb.gradient_fields(np.c_[X.ravel(), Y.ravel()], 4)
    n = X.size
    h = x[1] - x[0]
    for k in range(G.shape[1]):
        gx, gy = G[:n, k].reshape(X.shape), G[n:, k].reshape(X.shape)
        curl = np.gradient(gy, h, axis=0) - np.gradient(gx, h, axis=1)
        assert np.abs(curl[2:-2, 2:-2]).max() < 1e-3 * max(1, np.abs(G[:, k]).max())


def test_membrane_response_satisfies_free_edges_for_uniform_strain():
    # a uniform eigenstrain on a free plate is stress-free: response = e * (x, y) exactly
    E = pb.eigen_basis(tuple(map(tuple, P)), 2)
    assert np.allclose(E[:, 0], np.r_[P[:, 0], P[:, 1]], atol=1e-12)


def test_projection_recovers_magnification():
    rng = np.random.default_rng(0)
    df = _df(5, rng)
    m = rng.uniform(-20, 20, 5)
    F = np.outer(m * df["L_die"].to_numpy() / 2, np.r_[P[:, 0], P[:, 1]])
    C, rel = pb.project(FieldSet(df, F, 11))
    assert np.allclose(C[:, 0], m, atol=1e-8)
    assert rel.max() < 1e-12


def test_analytic_fields_match_analytic_magnification():
    df = _df(3, np.random.default_rng(1))
    F = pb.analytic_fields(df, P)
    C, _ = pb.project(FieldSet(df, F, 11))
    assert np.allclose(C[:, 0], pb.analytic_m(df))


def test_pinn_analytic_solution_has_zero_residual():
    from hbsim.pinn import ParametricPINN, physics_losses
    model = ParametricPINN(16, width=16, depth=2)
    for net in (model.net_u, model.net_e, model.net_phi):
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
    n = 64
    xy = 2 * torch.rand(n, 2) - 1
    p = torch.rand(n, 16)
    m = 10 * torch.randn(n, 1)
    xb = torch.tensor([[1.0, 0.3]] * n)
    nb = torch.tensor([[1.0, 0.0]] * n)
    L_pde, L_bc = physics_losses(model, xy, p, m, xb, p, m, nb)
    assert L_pde.item() < 1e-10 and L_bc.item() < 1e-10


def test_pinn_trains_and_predicts_shapes():
    from hbsim.pinn import PINNModel
    rng = np.random.default_rng(2)
    df = _df(4, rng)
    F = pb.analytic_fields(df, P) + rng.normal(0, 0.1, (4, 2 * len(P)))
    model = PINNModel(steps=20, width=16, depth=2, n_data=64, n_col=64, n_bnd=32, n_param_pool=16).fit(
        FieldSet(df, F, 11))
    out = model.predict(df)
    assert out.shape == F.shape and np.all(np.isfinite(out))


@pytest.mark.parametrize("k", [3, 4, 5, 9, 16])
def test_mark_layouts_are_unique_and_inside(k):
    idx = pb.mark_layout(k, 21)
    assert len(set(idx)) == k and idx.min() >= 0 and idx.max() < 441

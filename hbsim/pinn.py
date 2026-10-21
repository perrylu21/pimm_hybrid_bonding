"""Parametric physics-informed neural network (soft-constraint PINN) for die-shift fields. Host only.

Reduced physics of the bonded die (Kirchhoff-Love kinematics). The in-plane displacement of the
die's bonding surface is a membrane part plus a bending part,

    u = u_m + grad(phi),        phi = -(t_die / 2) w   (w: plate deflection),

so the bending part is curl-free by construction (hard constraint). The membrane part must satisfy
plane-stress equilibrium with an isotropic in-plane eigenstrain e (soft constraint, residual loss):

    div sigma = 0  in the die,   sigma . n = 0  on its edges,
    sigma = (1 - nu) eps(u) + nu tr(eps(u)) I - (1 + nu) e I       (plane stress / (E / (1 - nu^2)))

on the reference square [-1, 1]^2, with u in ppm units (field_nm / (L/2)) and e the in-plane
eigenstrain in ppm. Two networks take (xi, eta, p) with p the 16 process parameters scaled to
[0, 1]:

    u_m(xi, eta; p) = m_a(p) * (xi, eta) + S * N_u(xi, eta, p)
    e(xi, eta; p)   = m_a(p)             + S * N_e(xi, eta, p)
    phi(xi, eta; p) =                      S * N_phi(xi, eta, p)

m_a is the closed-form analytic magnification (the exact solution for a uniform eigenstrain), so
the PINN starts from the analytic answer and learns the correction. The physics residual is
evaluated at random parameter points that have no FEM run: this is where the PINN gains over a
pure data fit when FEM runs are scarce. Rigid motion is fixed by a gauge (zero mean translation
and rotation), matching the overlay fields after bonder alignment.
"""
import time

import numpy as np
import torch

from hbsim.materials import SI
from hbsim.physbasis import analytic_m, encode

NU = SI["nu"]


def _mlp(d_in, d_out, width, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, d_out))
    return torch.nn.Sequential(*layers)


class ParametricPINN(torch.nn.Module):
    def __init__(self, n_params=16, width=128, depth=4, scale=30.0, use_prior=True, bending=True):
        super().__init__()
        self.net_u = _mlp(2 + n_params, 2, width, depth)
        self.net_e = _mlp(2 + n_params, 1, width, depth)
        self.net_phi = _mlp(2 + n_params, 1, width, depth) if bending else None
        self.S, self.use_prior = scale, use_prior

    def forward(self, xy, p, m_a):
        """Membrane u_m (n, 2) and eigenstrain e (n, 1) in ppm. xy (n, 2), p (n, 16), m_a (n, 1)."""
        z = torch.cat([xy, p], 1)
        prior = m_a if self.use_prior else torch.zeros_like(m_a)
        u = prior * xy + self.S * self.net_u(z)
        e = prior + self.S * self.net_e(z)
        return u, e

    def total(self, xy, p, m_a):
        """Bonding-surface displacement u_m + grad(phi) (n, 2) in ppm; xy must require grad."""
        u, _ = self(xy, p, m_a)
        if self.net_phi is None:
            return u
        phi = self.S * self.net_phi(torch.cat([xy, p], 1))
        return u + _grad(phi, xy)


def _grad(out, x):
    return torch.autograd.grad(out, x, torch.ones_like(out), create_graph=True)[0]


def _stress(model, xy, p, m_a):
    xy = xy.requires_grad_(True)
    u, e = model(xy, p, m_a)
    gu, gv = _grad(u[:, :1], xy), _grad(u[:, 1:], xy)
    exx, eyy, exy = gu[:, :1], gv[:, 1:], 0.5 * (gu[:, 1:] + gv[:, :1])
    tr = exx + eyy
    sxx = (1 - NU) * exx + NU * tr - (1 + NU) * e
    syy = (1 - NU) * eyy + NU * tr - (1 + NU) * e
    sxy = (1 - NU) * exy
    rot = 0.5 * (gv[:, :1] - gu[:, 1:])
    return u, sxx, syy, sxy, rot


def physics_losses(model, xy, p, m_a, xy_b, p_b, m_b, n_b):
    """Equilibrium residual, free-edge traction and gauge (mean translation/rotation per param set)."""
    _, sxx, syy, sxy, _ = _stress(model, xy, p, m_a)
    rx = _grad(sxx, xy)[:, :1] + _grad(sxy, xy)[:, 1:]
    ry = _grad(sxy, xy)[:, :1] + _grad(syy, xy)[:, 1:]
    L_pde = ((rx ** 2 + ry ** 2) / model.S ** 2).mean()
    _, bxx, byy, bxy, _ = _stress(model, xy_b, p_b, m_b)
    tx = bxx * n_b[:, :1] + bxy * n_b[:, 1:]
    ty = bxy * n_b[:, :1] + byy * n_b[:, 1:]
    L_bc = ((tx ** 2 + ty ** 2) / model.S ** 2).mean()
    return L_pde, L_bc


def gauge_loss(model, n_sets, n_pts, n_params, gen):
    """Mean translation and mean rotation over the die must vanish, for random parameter sets."""
    p = torch.rand(n_sets, 1, n_params, generator=gen).expand(n_sets, n_pts, n_params).reshape(-1, n_params)
    xy = (2 * torch.rand(n_sets * n_pts, 2, generator=gen) - 1)
    m = torch.zeros(len(p), 1)  # the analytic term is gauge-free; check the network part only
    xy = xy.requires_grad_(True)
    u = model.total(xy, p, m)
    gu, gv = _grad(u[:, :1], xy), _grad(u[:, 1:], xy)
    rot = 0.5 * (gv[:, :1] - gu[:, 1:])
    u = u.reshape(n_sets, n_pts, 2).mean(1)
    rot = rot.reshape(n_sets, n_pts).mean(1)
    return ((u ** 2).sum(1).mean() + (rot ** 2).mean()) / model.S ** 2


def _decode_pm(p_unit):
    """Analytic magnification at unit-cube parameter points."""
    import pandas as pd
    from hbsim.params import from_unit
    from hbsim.fielddata import INPUTS
    df = pd.DataFrame({n: [from_unit(n, float(u)) for u in p_unit[:, j]] for j, n in enumerate(INPUTS)})
    return analytic_m(df)


class PINNModel:
    """Train / predict wrapper with the same interface as PhysicsBasisGP (fit(FieldSet), predict(df))."""

    def __init__(self, w_pde=1.0, w_bc=1.0, w_gauge=1.0, w_data=1.0, use_prior=True, steps=4000,
                 lr=2e-3, width=64, depth=4, n_data=2048, n_col=1024, n_bnd=256, n_param_pool=4096,
                 bending=True, seed=0, log=None):
        self.cfg = dict(w_pde=w_pde, w_bc=w_bc, w_gauge=w_gauge, w_data=w_data, steps=steps, lr=lr,
                        n_data=n_data, n_col=n_col, n_bnd=n_bnd, n_param_pool=n_param_pool)
        self.use_prior, self.width, self.depth, self.seed, self.log = use_prior, width, depth, seed, log
        self.bending = bending

    def fit(self, fs):
        c = self.cfg
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.P = fs.P
        n_p = len(fs.P)
        X = encode(fs.df)
        Fn = fs.F / fs.half_L[:, None]                    # ppm units
        mA = analytic_m(fs.df)
        # data tensors: every (run, pad) pair
        run_i = np.repeat(np.arange(len(fs)), n_p)
        pad_i = np.tile(np.arange(n_p), len(fs))
        D_xy = torch.tensor(fs.P[pad_i], dtype=torch.float32)
        D_p = torch.tensor(X[run_i], dtype=torch.float32)
        D_m = torch.tensor(mA[run_i, None], dtype=torch.float32)
        D_u = torch.tensor(np.column_stack([Fn[run_i, pad_i], Fn[run_i, pad_i + n_p]]), dtype=torch.float32)
        # collocation parameter pool: the training runs plus random (unlabeled) parameter points
        pool = np.vstack([X, rng.random((c["n_param_pool"], X.shape[1]))])
        pool_m = np.concatenate([mA, _decode_pm(pool[len(X):])])
        Pool = torch.tensor(pool, dtype=torch.float32)
        Pool_m = torch.tensor(pool_m[:, None], dtype=torch.float32)

        self.model = ParametricPINN(X.shape[1], self.width, self.depth, use_prior=self.use_prior,
                                    bending=self.bending)
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["steps"], eta_min=c["lr"] / 50)
        t0 = time.time()
        self.history = []
        for step in range(c["steps"] + 1):
            opt.zero_grad()
            loss = torch.zeros(())
            parts = {}
            if c["w_data"] > 0:
                b = torch.randint(0, len(D_xy), (c["n_data"],), generator=gen)
                u = self.model.total(D_xy[b].clone().requires_grad_(True), D_p[b], D_m[b])
                parts["data"] = (((u - D_u[b]) / self.model.S) ** 2).mean()
                loss = loss + c["w_data"] * parts["data"]
            if c["w_pde"] > 0 or c["w_bc"] > 0:
                k = torch.randint(0, len(Pool), (c["n_col"],), generator=gen)
                xy = 2 * torch.rand(c["n_col"], 2, generator=gen) - 1
                kb = torch.randint(0, len(Pool), (c["n_bnd"],), generator=gen)
                s = 2 * torch.rand(c["n_bnd"], generator=gen) - 1
                side = torch.randint(0, 4, (c["n_bnd"],), generator=gen)
                xb = torch.where(side == 0, -1.0, torch.where(side == 1, 1.0, s))
                yb = torch.where(side == 2, -1.0, torch.where(side == 3, 1.0, s))
                xy_b = torch.stack([xb, yb], 1)
                n_b = torch.stack([torch.where(side == 0, -1.0, torch.where(side == 1, 1.0, 0.0)),
                                   torch.where(side == 2, -1.0, torch.where(side == 3, 1.0, 0.0))], 1)
                parts["pde"], parts["bc"] = physics_losses(self.model, xy, Pool[k], Pool_m[k],
                                                           xy_b, Pool[kb], Pool_m[kb], n_b)
                loss = loss + c["w_pde"] * parts["pde"] + c["w_bc"] * parts["bc"]
            if c["w_gauge"] > 0:
                parts["gauge"] = gauge_loss(self.model, 16, 64, X.shape[1], gen)
                loss = loss + c["w_gauge"] * parts["gauge"]
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0:
                rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()},
                       "t": time.time() - t0}
                self.history.append(rec)
                if self.log:
                    self.log(" ".join(f"{k} {v:.3g}" for k, v in rec.items()))
        self.model.eval()
        return self

    def predict_norm(self, df, batch=65536):
        X = encode(df)
        mA = analytic_m(df)
        n_p = len(self.P)
        out = np.zeros((len(df), 2 * n_p))
        xy = torch.tensor(self.P, dtype=torch.float32)
        per = max(1, batch // n_p)
        for i in range(0, len(df), per):
            j = slice(i, min(i + per, len(df)))
            k = j.stop - j.start
            p = torch.tensor(np.repeat(X[j], n_p, 0), dtype=torch.float32)
            m = torch.tensor(np.repeat(mA[j], n_p)[:, None], dtype=torch.float32)
            u = self.model.total(xy.repeat(k, 1).requires_grad_(True), p, m)
            u = u.detach().numpy().reshape(k, n_p, 2)
            out[j] = np.concatenate([u[:, :, 0], u[:, :, 1]], 1)
        return out

    def predict(self, df):
        """Fields in nm, shape (n, 2N)."""
        return self.predict_norm(df) * (df["L_die"].to_numpy(dtype=float)[:, None] / 2.0)

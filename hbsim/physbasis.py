"""Hard-constrained physics-informed field model ("physics basis"). Host only.

Reduced physics of the bonded die (Kirchhoff-Love kinematics): the in-plane displacement of the
bonding surface is a membrane part plus a bending part, u = u_m + grad(phi) with phi = -(t/2) w.

  membrane: a free plane-stress die loaded by an isotropic in-plane eigenstrain e(x, y)
            div sigma = 0,  sigma = C : (eps(u_m) - e I),  sigma . n = 0 on the die edges;
  bending:  a curl-free field grad(phi) (surface displacement of a bent plate).

On the reference square [-1, 1]^2 (xi = x / (L/2)) the membrane response to each eigenstrain
monomial xi^i eta^j is computed once with scikit-fem; the bending part is spanned by gradients of
polynomial potentials. Every field the model can produce is a combination of these columns plus
the three rigid modes, so the reduced physics holds exactly ("hard constraint"). Units: coefficients
in ppm; 1 ppm acting over L/2 [mm] moves a pad by (L/2) nm, so field_nm = (L/2) * B @ c.

Parametric model: the coefficients are compressed with PCA (in field space) and one Gaussian
process per retained component maps the 16 process parameters to the scores. The uniform
eigenstrain carries the closed-form analytic prior (materials.analytic_baseline), so with no data
the model already predicts the analytic magnification.
"""
import functools

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from hbsim.fielddata import INPUTS
from hbsim.materials import SI, analytic_baseline
from hbsim.params import to_unit


def encode(df):
    """(n, 16) unit-cube inputs, same scaling as hbsim.surrogate.encode."""
    return np.column_stack([[to_unit(n, v) for v in df[n].to_numpy(dtype=float)] for n in INPUTS])


def analytic_m(df):
    """Analytic isotropic magnification [ppm] per run: the prior on the uniform eigenstrain."""
    return np.array([analytic_baseline(r)["m_x"] for r in df[INPUTS].to_dict("records")])


@functools.lru_cache(maxsize=4)
def eigen_basis(P_key, deg=4, nu=SI["nu"], n_el=32):
    """Pad responses (2N, n_terms) to eigenstrain monomials of total degree <= deg.

    P_key: tuple of pad coordinates on the reference square (hashable for caching).
    Column 0 is the uniform eigenstrain, whose response is exactly (xi, eta).
    """
    from skfem import (Basis, BilinearForm, ElementQuad2, ElementVector, LinearForm, MeshQuad,
                       asm, solve)
    from skfem.helpers import ddot, dot, eye, sym_grad, trace

    P = np.array(P_key)
    mesh = MeshQuad.init_tensor(np.linspace(-1, 1, n_el + 1), np.linspace(-1, 1, n_el + 1))
    vb = Basis(mesh, ElementVector(ElementQuad2()))

    def sig(e):  # plane stress, scaled by E / (1 - nu^2)
        return (1 - nu) * e + nu * eye(trace(e), 2)

    @BilinearForm
    def a(u, v, w):  # tiny spring removes the rigid modes; they are fitted separately
        return ddot(sig(sym_grad(u)), sym_grad(v)) + 1e-9 * dot(u, v)

    A = asm(a, vb)
    (_, bx), (_, by) = vb.split(np.zeros(vb.N))
    Px, Py = bx.probes(P.T), by.probes(P.T)
    cols = []
    for k in range(deg + 1):
        for i in range(k + 1):
            j = k - i

            @LinearForm
            def l(v, w, i=i, j=j):
                return (1 + nu) * (w.x[0] ** i * w.x[1] ** j) * trace(sym_grad(v))

            u = solve(A, asm(l, vb))
            (ux, _), (uy, _) = vb.split(u)
            cols.append(np.concatenate([Px @ ux, Py @ uy]))
    B = np.array(cols).T
    B[:, 0] = np.concatenate([P[:, 0], P[:, 1]])  # exact response to a uniform eigenstrain
    return B


def rigid_modes(P):
    """Translation x, translation y, rotation on the reference square, shape (2N, 3)."""
    x, y = P[:, 0], P[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    return np.column_stack([np.r_[o, z], np.r_[z, o], np.r_[-y, x]])


def affine_modes(P):
    """Industry 6-parameter overlay model: translation, rotation/skew and magnification per axis."""
    x, y = P[:, 0], P[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    return np.column_stack([np.r_[o, z], np.r_[z, o], np.r_[x, z], np.r_[z, y], np.r_[-y, z], np.r_[z, x]])


def gradient_fields(P, deg=4):
    """Curl-free fields grad(xi^i eta^j), 2 <= i + j <= deg (bending part), shape (2N, n)."""
    x, y = P[:, 0], P[:, 1]
    cols = []
    for k in range(2, deg + 1):
        for i in range(k + 1):
            j = k - i
            gx = i * x ** (i - 1) * y ** j if i > 0 else np.zeros_like(x)
            gy = j * x ** i * y ** (j - 1) if j > 0 else np.zeros_like(y)
            cols.append(np.concatenate([gx, gy]))
    return np.array(cols).T


def _orth(W, tol=1e-8):
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, S > tol * S[0]] if len(S) else W


def full_basis(P, deg=4, bending=True, deg_bend=4):
    """[uniform eigenstrain | orthonormal non-uniform physics columns | rigid modes].

    Column 0 is the exact response to a uniform eigenstrain, (xi, eta), so its coefficient is the
    isotropic magnification in ppm. The non-uniform membrane and bending columns are
    orthogonalized against it and the rigid modes, and nearly dependent directions are dropped:
    without this, almost collinear columns let a least-squares fit "explain" anything with huge
    coefficients.
    """
    E = eigen_basis(tuple(map(tuple, P)), deg)
    c0, R = E[:, :1], rigid_modes(P)
    W = E[:, 1:]
    if bending:
        W = np.column_stack([W, gradient_fields(P, deg_bend)])
    Q0 = _orth(np.column_stack([c0, R]))
    W = W - Q0 @ (Q0.T @ W)
    return np.column_stack([c0, _orth(W), R])


def project(fs, deg=4, bending=True):
    """Least-squares coefficients of each field in the physics basis (ppm units), shape (n, k).

    Returns (coefficients, relative residual energy of the projection per run).
    """
    B = full_basis(fs.P, deg, bending)
    Fn = fs.F / fs.half_L[:, None]
    C, *_ = np.linalg.lstsq(B, Fn.T, rcond=None)
    R = Fn - (B @ C).T
    return C.T, (R ** 2).sum(1) / np.maximum((Fn ** 2).sum(1), 1e-30)


def _gp(seed):
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(INPUTS)), nu=2.5,
                                     length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-4, (1e-9, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=seed)


class PhysicsBasisGP:
    """Parametric field model: analytic prior + GPs on PCA scores of physics-basis coefficients."""

    def __init__(self, deg=4, energy=0.999, max_modes=8, use_prior=True, bending=True, seed=0):
        self.deg, self.energy, self.max_modes, self.use_prior, self.seed = deg, energy, max_modes, use_prior, seed
        self.bending = bending

    def true_scores(self, fs):
        """Coordinates of fields on modes(): uniform coefficient [ppm] and PCA scores."""
        C, _ = project(fs, self.deg, self.bending)
        Cn = C.copy()
        Cn[:, 0] = 0.0
        Cn[:, -3:] = 0.0                                  # rigid motion: removed by the overlay fit
        Y = Cn @ self.B.T                                 # non-uniform part, inside the physics span
        return C[:, 0], Y

    def fit(self, fs, modes_from=None):
        """Fit the GPs. modes_from: another PhysicsBasisGP whose PCA modes are reused (CV folds)."""
        self.P = fs.P
        self.B = full_basis(fs.P, self.deg, self.bending)
        c0, Y = self.true_scores(fs)
        if self.use_prior:
            c0 = c0 - analytic_m(fs.df)
        X = encode(fs.df)
        # Uniform eigenstrain (magnification) gets its own GP: it carries > 95% of the field energy
        # and would otherwise crowd the non-uniform modes out of the PCA.
        self.gp_uniform = _gp(self.seed).fit(X, c0)
        if modes_from is None:
            self.mean = Y.mean(0)
            U, S, Vt = np.linalg.svd(Y - self.mean, full_matrices=False)
            frac = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-30)
            r = int(min(self.max_modes, np.searchsorted(frac, self.energy) + 1, len(S)))
            self.V = Vt[:r]
        else:
            self.mean, self.V = modes_from.mean, modes_from.V
        scores = (Y - self.mean) @ self.V.T
        self.gps = [_gp(self.seed + 1 + k).fit(X, scores[:, k]) for k in range(len(self.V))]
        return self

    def score_targets(self, fs):
        """(n, 1 + r) true coordinates comparable with predict_scores()."""
        c0, Y = self.true_scores(fs)
        return np.column_stack([c0, (Y - self.mean) @ self.V.T])

    def modes(self):
        """Columns spanning the model's field space (ppm units): uniform, PCA modes."""
        return np.column_stack([self.B[:, 0], self.V.T])

    def predict_scores(self, df, return_std=False):
        """(n, 1 + r) predicted coefficients on modes() (uniform includes the analytic prior)."""
        X = encode(df)
        gps = [self.gp_uniform] + self.gps
        out = [g.predict(X, return_std=True) for g in gps]
        s = np.column_stack([o[0] for o in out])
        if self.use_prior:
            s[:, 0] += analytic_m(df)
        if return_std:
            return s, np.column_stack([o[1] for o in out])
        return s

    def predict_norm(self, df):
        """Fields in ppm units ((n, 2N): divide-by-(L/2) form)."""
        return self.mean + self.predict_scores(df) @ self.modes().T

    def predict(self, df):
        """Fields in nm, shape (n, 2N)."""
        return self.predict_norm(df) * (df["L_die"].to_numpy(dtype=float)[:, None] / 2.0)


def analytic_fields(df, P):
    """Zero-data prediction: the analytic isotropic magnification field [nm]."""
    m = analytic_m(df)
    xy = np.concatenate([P[:, 0], P[:, 1]])
    return np.outer(m * df["L_die"].to_numpy(dtype=float) / 2.0, xy)


# ---------------------------------------------------------------------------------------------
# Sparse overlay-mark reconstruction (experiment E2)
# ---------------------------------------------------------------------------------------------

def mark_layout(k, n_pads=21):
    """Pad indices (into the n_pads x n_pads grid, row-major in x) of k overlay marks."""
    e = n_pads - 1
    c = e // 2
    layouts = {
        3: [(0, 0), (e, 0), (c, e)],
        4: [(0, 0), (0, e), (e, 0), (e, e)],
        5: [(0, 0), (0, e), (e, 0), (e, e), (c, c)],
        9: [(i, j) for i in (0, c, e) for j in (0, c, e)],
        16: [(i, j) for i in (0, e // 3, 2 * e // 3, e) for j in (0, e // 3, 2 * e // 3, e)],
    }
    if k not in layouts:
        raise ValueError(f"no mark layout for k={k}; choose from {sorted(layouts)}")
    return np.array([i * n_pads + j for i, j in layouts[k]])


def _rows(marks, n):
    return np.concatenate([marks, marks + n])


def reconstruct_affine(P, marks, meas_nm):
    """Industry practice: fit the 6-parameter overlay model to the marks, evaluate at every pad."""
    A = affine_modes(P)
    c, *_ = np.linalg.lstsq(A[_rows(marks, len(P))], meas_nm, rcond=None)
    return A @ c


def polynomial_modes(P, order):
    """Higher-order overlay correction: monomials x^i y^j (i + j <= order) for each axis separately.

    Shape (2N, (order+1)(order+2)); order 1 is the 6-parameter affine model in another basis.
    """
    x, y = P[:, 0], P[:, 1]
    z = np.zeros_like(x)
    mono = [x ** i * y ** (k - i) for k in range(order + 1) for i in range(k + 1)]
    return np.column_stack([np.r_[m, z] for m in mono] + [np.r_[z, m] for m in mono])


def reconstruct_polynomial(P, marks, meas_nm, order):
    """Empirical baseline: fit an order-`order` polynomial correction per axis to the marks."""
    terms = (order + 1) * (order + 2) // 2
    if len(marks) < terms:
        raise ValueError(f"order-{order} correction needs >= {terms} marks per axis, got {len(marks)} marks")
    A = polynomial_modes(P, order)
    c, *_ = np.linalg.lstsq(A[_rows(marks, len(P))], meas_nm, rcond=None)
    return A @ c


def reconstruct_prior_affine(P, marks, meas_nm, prior_nm):
    """Baseline with the same prior as the physics method: prior field + affine fit to the remainder."""
    rows = _rows(marks, len(P))
    A = affine_modes(P)
    c, *_ = np.linalg.lstsq(A[rows], meas_nm - prior_nm[rows], rcond=None)
    return prior_nm + A @ c


def reconstruct_physics(P, marks, meas_nm, half_L, prior_mean_norm, prior_cov, modes, noise_nm):
    """Bayesian (MAP) reconstruction inside the physics span.

    Field (ppm units) = prior_mean + modes @ z, with z ~ N(0, prior_cov); modes include the rigid
    motions (broad prior) and the parametric model's PCA modes (calibrated prior). meas_nm are the
    measured overlays at the marks [nm].
    """
    rows = _rows(marks, len(P))
    A = modes[rows] * half_L
    r = meas_nm - prior_mean_norm[rows] * half_L
    S = prior_cov
    K = S @ A.T @ np.linalg.inv(A @ S @ A.T + noise_nm ** 2 * np.eye(len(rows)))
    z = K @ r
    return (prior_mean_norm + modes @ z) * half_L

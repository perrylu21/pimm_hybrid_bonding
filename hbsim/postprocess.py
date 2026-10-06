"""Overlay fitting and metrics. Pure numpy (host and container)."""
import numpy as np

OUTPUTS = ["t_x", "t_y", "theta", "m_x", "m_y", "resid_3sigma", "d_max"]


def pad_grid(L, n):
    """n x n pad positions [mm] covering the die interface, inset 2% from the edges."""
    s = np.linspace(-0.49 * L, 0.49 * L, n)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def fit_overlay(xy, d):
    """Least-squares fit d = t + theta*(-y, x) + diag(m_x, m_y)*(x, y) + r.

    xy, d: (N, 2) arrays in mm. Returns dict with t_x, t_y [mm], theta [rad], m_x, m_y [-],
    and resid (N, 2) [mm].
    """
    x, y = xy[:, 0], xy[:, 1]
    N = len(x)
    A = np.zeros((2 * N, 5))
    A[:N, 0] = 1.0
    A[:N, 2] = -y
    A[:N, 3] = x
    A[N:, 1] = 1.0
    A[N:, 2] = x
    A[N:, 4] = y
    b = np.concatenate([d[:, 0], d[:, 1]])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    return {"t_x": c[0], "t_y": c[1], "theta": c[2], "m_x": c[3], "m_y": c[4],
            "resid": np.column_stack([r[:N], r[N:]])}


def remove_rigid(xy, d, fit):
    """Subtract the translation and rotation of a fit (bonder alignment) from overlay d."""
    t = np.array([fit["t_x"], fit["t_y"]])
    rot = fit["theta"] * np.column_stack([-xy[:, 1], xy[:, 0]])
    return d - t - rot


def metrics(xy, d):
    """Paper metrics from overlay vectors in mm: lengths in nm, rotation in urad, magnification in ppm."""
    f = fit_overlay(xy, d)
    return {
        "t_x": float(f["t_x"]) * 1e6,
        "t_y": float(f["t_y"]) * 1e6,
        "theta": float(f["theta"]) * 1e6,
        "m_x": float(f["m_x"]) * 1e6,
        "m_y": float(f["m_y"]) * 1e6,
        "resid_3sigma": 3.0 * float(np.std(f["resid"])) * 1e6,
        "d_max": float(np.max(np.linalg.norm(d, axis=1))) * 1e6,
    }


def apply_placement(xy, d, dx_nm, dy_nm, dtheta_urad):
    """Add bonder placement offset [nm] and rotation error [urad] to overlay d [mm]."""
    rot = dtheta_urad * 1e-6 * np.column_stack([-xy[:, 1], xy[:, 0]])
    return d + np.array([dx_nm, dy_nm]) * 1e-6 + rot

"""Thermo-mechanical FEM of the D2W hybrid-bonding moment (runs inside the DOLFINx container).

Step 1: transient heat conduction on the full die + wafer mesh.
Step 2: quasi-static thermo-elasticity of die and wafer as separate bodies (submeshes),
        one LU factorization per body reused for every time step.
Step 3: bond-wave locking -> overlay vectors at the pad grid.
See docs/superpowers/specs/2026-10-05-hybrid-bonding-die-shift-design.md, section 3.
"""
import gmsh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh as dmesh
from dolfinx.io import gmsh as gmshio
import dolfinx.fem.petsc as fpetsc

from hbsim import geometry as geom
from hbsim import postprocess
from hbsim.materials import SI, T_REF, bond_layer, curvature, to_model

COMM = MPI.COMM_SELF  # each run is serial; parallelism comes from running many runs at once
TOL = 1e-9
T_FIRST = 1e-6  # s, first thermal step after contact (below the ~1e-5 s die diffusion time)
NUMERIC_KEYS = ("mesh_n", "n_ring", "nz_die", "nz_wafer", "n_t", "margin_frac", "deg", "n_pads")


def build_mesh(m, num):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geom.build_model(m, num)
        gmsh.model.mesh.generate(3)
        mesh_data = gmshio.model_to_mesh(gmsh.model, COMM, 0, gdim=3)
    finally:
        gmsh.finalize()
    return mesh_data.mesh


def _layered(msh, si_value, bond_value, t_bond):
    """DG0 field: bond_value in the bond layers (cell midpoint |z| < t_bond), si_value elsewhere."""
    f = fem.Function(fem.functionspace(msh, ("DG", 0)))
    f.interpolate(lambda x: np.where(np.abs(x[2]) < t_bond, bond_value, si_value))
    return f


def _lu_solver(A):
    ksp = PETSc.KSP().create(COMM)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    return ksp


def solve_thermal(msh, m, times):
    """Backward-Euler heat conduction. Returns (P1 space, list of T arrays, one per time)."""
    lv = geom.levels(m)
    bl = bond_layer(m)
    V = fem.functionspace(msh, ("Lagrange", 1))
    k = _layered(msh, SI["k"], bl["k"], m["t_bond"])
    rc = _layered(msh, SI["rho_cp"], bl["rho_cp"], m["t_bond"])
    T_n = fem.Function(V)
    T_n.interpolate(lambda x: np.where(x[2] > TOL, m["T_head"], m["T_chuck"]))
    fdim = msh.topology.dim - 1
    top = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["die_top"]))
    bot = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["wafer_bottom"]))
    bcs = [fem.dirichletbc(PETSc.ScalarType(m["T_head"]), fem.locate_dofs_topological(V, fdim, top), V),
           fem.dirichletbc(PETSc.ScalarType(m["T_chuck"]), fem.locate_dofs_topological(V, fdim, bot), V)]
    dt = fem.Constant(msh, PETSc.ScalarType(times[1] - times[0]))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(rc * u * v * ufl.dx + dt * k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(rc * T_n * v * ufl.dx)
    A = fpetsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    ksp = _lu_solver(A)
    T = fem.Function(V)
    history = [T_n.x.array.copy()]
    for t_prev, t_next in zip(times[:-1], times[1:]):
        if not np.isclose(t_next - t_prev, float(dt.value)):  # graded grid: re-factor for new dt
            dt.value = t_next - t_prev
            A.zeroEntries()
            fpetsc.assemble_matrix(A, a, bcs=bcs)
            A.assemble()
            ksp.setOperators(A)
        b = fpetsc.assemble_vector(L)
        fpetsc.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, bcs)
        ksp.solve(b, T.x.petsc_vec)
        T.x.scatter_forward()
        T_n.x.array[:] = T.x.array
        history.append(T.x.array.copy())
        b.destroy()
    ksp.destroy()
    A.destroy()
    return V, history


def _component_bc(V, comp, marker, value_fn):
    """Dirichlet BC on one displacement component at dofs whose coordinates satisfy marker."""
    Vc, _ = V.sub(comp).collapse()
    dofs = fem.locate_dofs_geometrical((V.sub(comp), Vc), marker)
    g = fem.Function(Vc)
    g.interpolate(value_fn)
    return fem.dirichletbc(g, dofs, V.sub(comp))


def _at(px, py, pz):
    return lambda x: np.isclose(x[0], px) & np.isclose(x[1], py) & np.isclose(x[2], pz)


def _zero(x):
    return np.zeros(x.shape[1])


class ElasticBody:
    """Linear thermo-elastic body on a submesh with a reusable LU factorization."""

    def __init__(self, parent, cells, m, num, which):
        self.mesh, emap, _, _ = dmesh.create_submesh(parent, parent.topology.dim, cells)
        n_sub = self.mesh.topology.index_map(3).size_local
        self.sub_cells = np.arange(n_sub, dtype=np.int32)
        self.parent_cells = emap.sub_topology_to_topology(self.sub_cells, inverse=False)
        msh = self.mesh
        lv = geom.levels(m)
        bl = bond_layer(m)
        V = fem.functionspace(msh, ("Lagrange", int(num["deg"]), (3,)))
        E = _layered(msh, SI["E"], bl["E"], m["t_bond"])
        nu = _layered(msh, SI["nu"], bl["nu"], m["t_bond"])
        alpha = _layered(msh, SI["alpha"], bl["alpha"], m["t_bond"])
        mu = E / (2 * (1 + nu))
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        I = ufl.Identity(3)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(e):
            return 2 * mu * e + lmbda * ufl.tr(e) * I

        self.T = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
        eig = alpha * (self.T - T_REF) * I
        x = ufl.SpatialCoordinate(msh)
        fdim = 2
        p = m["pressure"]
        if which == "die":
            # Bow: compatible (stress-free) through-thickness eigenstrain over the whole die body.
            e_bow = -curvature(m) * (x[2] - lv["die_top"] / 2.0)
            eig = eig + ufl.as_matrix([[e_bow, 0, 0], [0, e_bow, 0], [0, 0, 0]])
            # Rigid modes: in-plane ones by statically determinate point pins on the interface
            # plane. With k_head > 0 the head spring already removes the out-of-plane modes;
            # extra u_z pins would over-constrain the die and create spurious reactions.
            L = m["L"]
            bcs = [_component_bc(V, 0, _at(0, 0, 0), _zero),
                   _component_bc(V, 1, _at(0, 0, 0), _zero),
                   _component_bc(V, 1, _at(L / 2, 0, 0), _zero)]
            if m["k_head"] == 0:  # exactly free die (verification): pin the 3 out-of-plane modes
                bcs += [_component_bc(V, 2, _at(0, 0, 0), _zero),
                        _component_bc(V, 2, _at(L / 2, 0, 0), _zero),
                        _component_bc(V, 2, _at(0, L / 2, 0), _zero)]
            # Bond force: equal and opposite pressure on die top (tag 1) and bottom (tag 2).
            top = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], lv["die_top"]))
            bot = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], 0.0))
            traction = [(top, (0.0, 0.0, -p)), (bot, (0.0, 0.0, p))]
        elif which == "wafer":
            W2 = m["L"] / 2 + num["margin_frac"] * m["L"]
            dT = m["T_chuck"] - T_REF
            a_c, a_s = m["alpha_chuck"] * dT, SI["alpha"] * dT
            zb = lv["wafer_bottom"]

            def on_bottom(x):
                return np.isclose(x[2], zb)

            def on_sides(x):  # bottom-edge nodes belong to the bottom BC
                return ((np.isclose(np.abs(x[0]), W2) | np.isclose(np.abs(x[1]), W2))
                        & (x[2] > zb + TOL))

            bcs = [_component_bc(V, 0, on_bottom, lambda x: a_c * x[0]),
                   _component_bc(V, 1, on_bottom, lambda x: a_c * x[1]),
                   _component_bc(V, 2, on_bottom, _zero),
                   _component_bc(V, 0, on_sides, lambda x: a_s * x[0]),
                   _component_bc(V, 1, on_sides, lambda x: a_s * x[1])]
            h = m["L"] / 2 + TOL
            foot = dmesh.locate_entities_boundary(
                msh, fdim, lambda x: np.isclose(x[2], 0.0) & (np.abs(x[0]) <= h) & (np.abs(x[1]) <= h))
            traction = [(foot, (0.0, 0.0, -p))]
        else:
            raise ValueError(f"which must be 'die' or 'wafer', got {which!r}")

        facets = np.concatenate([f for f, _ in traction]).astype(np.int32)
        markers = np.concatenate([np.full(len(f), i + 1, dtype=np.int32) for i, (f, _) in enumerate(traction)])
        order = np.argsort(facets)
        ft = dmesh.meshtags(msh, fdim, facets[order], markers[order])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        lhs = ufl.inner(sigma(eps(u)), eps(v)) * ufl.dx
        if which == "die":
            # Bond-head compliance: normal spring on the die top. k_head -> 0 is a free die,
            # k_head -> infinity is a rigid flat head that flattens the bow.
            lhs = lhs + m["k_head"] * u[2] * v[2] * ds(1)
        rhs = ufl.inner(sigma(eig), eps(v)) * ufl.dx
        for i, (_, t) in enumerate(traction):
            rhs = rhs + ufl.dot(fem.Constant(msh, np.array(t, dtype=PETSc.ScalarType)), v) * ds(i + 1)
        self.a = fem.form(lhs)
        self.L = fem.form(rhs)
        self.bcs = bcs
        self.A = fpetsc.assemble_matrix(self.a, bcs=bcs)
        self.A.assemble()
        self.ksp = _lu_solver(self.A)
        self.u = fem.Function(V)
        self.V = V
        self._tree = geometry.bb_tree(msh, 3)

    def solve(self, T_parent=None, T_uniform=None):
        """Solve with a parent-mesh P1 temperature Function, or a uniform temperature."""
        if T_uniform is not None:
            self.T.x.array[:] = T_uniform
        else:
            self.T.interpolate(T_parent, cells0=self.parent_cells, cells1=self.sub_cells)
        b = fpetsc.assemble_vector(self.L)
        fpetsc.apply_lifting(b, [self.a], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, self.bcs)
        self.ksp.solve(b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        b.destroy()
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"elastic solve failed, KSP reason {reason}")

    def eval_xy(self, xy):
        """In-plane displacement (N, 2) [mm] at interface points (x, y, 0)."""
        P = np.column_stack([xy, np.zeros(len(xy))])
        coll = geometry.compute_collisions_points(self._tree, P)
        cc = geometry.compute_colliding_cells(self.mesh, coll, P)
        cells = []
        for i in range(len(P)):
            links = cc.links(i)
            if len(links) == 0:
                raise RuntimeError(f"pad point {P[i]} not found in mesh")
            cells.append(links[0])
        return self.u.eval(P, np.array(cells, dtype=np.int32))[:, :2]

    @property
    def n_dofs(self):
        return self.V.dofmap.index_map.size_global * 3

    def destroy(self):
        self.ksp.destroy()
        self.A.destroy()


def simulate(p, lock="wave"):
    """Run one simulation from a resolved parameter dict (user units). Does not validate ranges.

    lock="wave": pads lock at bond-wave arrival times (production).
    lock="t0":   all pads lock at contact, t = 0 (verification: uniform initial temperatures).
    Returns dict with pads (N,2) [mm], d (N,2) overlay [mm] after bonder alignment, n_dofs.
    """
    if lock not in ("wave", "t0"):
        raise ValueError(f"lock must be 'wave' or 't0', got {lock!r}")
    m = to_model(p)
    num = {k: p[k] for k in NUMERIC_KEYS}
    msh = build_mesh(m, num)
    pads = postprocess.pad_grid(m["L"], int(num["n_pads"]))
    die = ElasticBody(msh, dmesh.locate_entities(msh, 3, lambda x: x[2] >= -TOL), m, num, "die")
    waf = ElasticBody(msh, dmesh.locate_entities(msh, 3, lambda x: x[2] <= TOL), m, num, "wafer")
    try:
        # t = 0: pre-contact state, each body at its own uniform temperature.
        die.solve(T_uniform=m["T_head"])
        waf.solve(T_uniform=m["T_chuck"])
        d0 = die.eval_xy(pads) - waf.eval_xy(pads)
        align = postprocess.fit_overlay(pads, d0)  # bonder alignment removes this rigid motion
        if lock == "t0":
            d = postprocess.remove_rigid(pads, d0, align)
        else:
            origin = np.array([m["x0"], m["y0"]])
            corners = np.array([[sx * m["L"] / 2, sy * m["L"] / 2] for sx in (-1, 1) for sy in (-1, 1)])
            t_end = np.max(np.linalg.norm(corners - origin, axis=1)) / m["v_wave"]
            # Graded grid: the thermal transient after contact is ~1e-5-1e-2 s while the wave takes
            # ~0.1 s, so steps start at T_FIRST and grow geometrically to t_end.
            times = np.concatenate([[0.0], np.geomspace(min(T_FIRST, t_end / 10), t_end, int(num["n_t"]))])
            Vt, T_hist = solve_thermal(msh, m, times)
            T_par = fem.Function(Vt)
            diffs = [d0]
            for Tk in T_hist[1:]:
                T_par.x.array[:] = Tk
                die.solve(T_parent=T_par)
                waf.solve(T_parent=T_par)
                diffs.append(die.eval_xy(pads) - waf.eval_xy(pads))
            diffs = np.array(diffs)  # (n_t+1, N, 2)
            t_lock = np.linalg.norm(pads - origin, axis=1) / m["v_wave"]
            d_lock = np.array([[np.interp(t_lock[i], times, diffs[:, i, c]) for c in range(2)]
                               for i in range(len(pads))])
            d = postprocess.remove_rigid(pads, d_lock, align)
        if not np.all(np.isfinite(d)):
            raise RuntimeError("non-finite overlay values")
        return {"pads": pads, "d": d, "n_dofs": die.n_dofs + waf.n_dofs}
    finally:
        die.destroy()
        waf.destroy()

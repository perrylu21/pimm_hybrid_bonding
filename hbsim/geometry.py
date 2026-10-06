"""Gmsh model of the die + bond layers + wafer patch as a structured hexahedral mesh.

Layout (model units, mm): interface plane z = 0. Die body (bond layer + Si) occupies
[-L/2, L/2]^2 x [0, t_bond + t_die]; wafer body occupies [-W/2, W/2]^2 x [-(t_bond + t_wafer), 0]
with W = L + 2*margin. Physical volumes: 1 = die body, 2 = wafer body.
"""
import gmsh


def levels(m):
    """z coordinates of the layer interfaces."""
    tb = m["t_bond"]
    return {"die_top": tb + m["t_die"], "die_bond_top": tb, "interface": 0.0,
            "wafer_bond_bottom": -tb, "wafer_bottom": -(tb + m["t_wafer"])}


def build_model(m, num):
    """Create the gmsh geometry and mesh constraints in the current gmsh session (no meshing)."""
    L = m["L"]
    W = L + 2 * num["margin_frac"] * L
    tb, td, tw = m["t_bond"], m["t_die"], m["t_wafer"]
    geo = gmsh.model.geo
    xs = [-W / 2, -L / 2, L / 2, W / 2]
    nseg = [int(num["n_ring"]), int(num["mesh_n"]), int(num["n_ring"])]
    pts = {(i, j): geo.addPoint(xs[i], xs[j], 0.0) for i in range(4) for j in range(4)}
    hl = {(i, j): geo.addLine(pts[i, j], pts[i + 1, j]) for j in range(4) for i in range(3)}
    vl = {(i, j): geo.addLine(pts[i, j], pts[i, j + 1]) for i in range(4) for j in range(3)}
    surf = {}
    for i in range(3):
        for j in range(3):
            loop = geo.addCurveLoop([hl[i, j], vl[i + 1, j], -hl[i, j + 1], -vl[i, j]])
            surf[i, j] = geo.addPlaneSurface([loop])
    # Structured-mesh constraints must be set on the geo kernel before synchronize,
    # otherwise extrusion produces prisms instead of hexahedra.
    for (i, _), tag in hl.items():
        geo.mesh.setTransfiniteCurve(tag, nseg[i] + 1)
    for (_, j), tag in vl.items():
        geo.mesh.setTransfiniteCurve(tag, nseg[j] + 1)
    for s in surf.values():
        geo.mesh.setTransfiniteSurface(s)
        geo.mesh.setRecombine(2, s)
    wafer_vols = []
    for s in surf.values():
        out = geo.extrude([(2, s)], 0, 0, -(tb + tw), numElements=[1, int(num["nz_wafer"])],
                          heights=[tb / (tb + tw), 1.0], recombine=True)
        wafer_vols += [t for d, t in out if d == 3]
    out = geo.extrude([(2, surf[1, 1])], 0, 0, tb + td, numElements=[1, int(num["nz_die"])],
                      heights=[tb / (tb + td), 1.0], recombine=True)
    die_vols = [t for d, t in out if d == 3]
    geo.synchronize()
    gmsh.model.addPhysicalGroup(3, die_vols, 1, name="die")
    gmsh.model.addPhysicalGroup(3, wafer_vols, 2, name="wafer")

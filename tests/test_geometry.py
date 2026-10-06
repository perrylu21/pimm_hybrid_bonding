import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
from hbsim import geometry, materials, params  # noqa: E402


def _mesh(**overrides):
    p = {**params.defaults(), **overrides}
    m = materials.to_model(p)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geometry.build_model(m, p)
        gmsh.model.mesh.generate(3)
        types, tags, _ = gmsh.model.mesh.getElements(3)
        _, coords, _ = gmsh.model.mesh.getNodes()
    finally:
        gmsh.finalize()
    return p, m, types, tags, coords.reshape(-1, 3)


def test_all_hexahedra_with_expected_count():
    p, _, types, tags, _ = _mesh(mesh_n=8, n_ring=2, nz_die=2, nz_wafer=2)
    assert list(types) == [5]  # gmsh type 5 = 8-node hexahedron
    n_die = 8 * 8 * (1 + 2)
    n_wafer = (8 + 2 * 2) ** 2 * (1 + 2)
    assert len(tags[0]) == n_die + n_wafer


def test_key_nodes_exist_for_pins_and_layers():
    p, m, _, _, X = _mesh(mesh_n=8, n_ring=2, nz_die=2, nz_wafer=2, t_bond=0.05)
    lv = geometry.levels(m)
    L = m["L"]

    def has(pt):
        return np.any(np.all(np.isclose(X, pt, atol=1e-9), axis=1))  # gmsh places nodes to ~1e-11

    assert has((0, 0, 0)) and has((L / 2, 0, 0)) and has((0, L / 2, 0))
    for z in lv.values():
        assert np.any(np.isclose(X[:, 2], z, atol=1e-9)), z

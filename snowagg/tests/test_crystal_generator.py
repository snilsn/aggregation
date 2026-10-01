"""Crystal dimensions and generated lattices are bit-identical to the
Python reference for all crystal types."""
import numpy as np
import pytest
from aggregation import crystal as rc
from aggregation import generator as rg

from conftest import NullRot
from snowagg import _core

TAN_ALPHA = np.tan(28 * (np.pi / 180.0))  # numpy's value, see crystal.hpp

SIZES = np.exp(np.linspace(np.log(20e-6), np.log(3e-3), 11))


def make_pair(kind, D, dendrite_grid):
    if kind == "plate":
        return rc.Plate(D), _core.Plate(D)
    if kind == "column":
        return rc.Column(D), _core.Column(D)
    if kind == "needle":
        return rc.Needle(D), _core.Needle(D)
    if kind == "rosette":
        return rc.Rosette(D), _core.Rosette(D, TAN_ALPHA)
    if kind == "bullet":
        return rc.Bullet(D), _core.Bullet(D, TAN_ALPHA)
    if kind == "spheroid":
        return rc.Spheroid(D, 0.6), _core.Spheroid(D, 0.6)
    if kind == "dendrite":
        return rc.Dendrite(D, hex_grid=dendrite_grid), _core.Dendrite(D, dendrite_grid)
    raise ValueError(kind)


KINDS = ["plate", "column", "needle", "rosette", "bullet", "spheroid", "dendrite"]


@pytest.mark.parametrize("kind", KINDS)
def test_dimensions(kind, dendrite_grid):
    for D in SIZES:
        py, cc = make_pair(kind, D, dendrite_grid)
        assert py.max_radius() == cc.max_radius()
        attrs = {"plate": "a L V A r", "column": "a L V A r", "needle": "a L V A r",
                 "dendrite": "a L V A r grid_D", "rosette": "L a t", "bullet": "L a t",
                 "spheroid": "a c"}[kind]
        for name in attrs.split():
            assert getattr(py, name) == getattr(cc, name), (kind, D, name)


@pytest.mark.parametrize("kind", KINDS)
def test_lattice_identical(kind, dendrite_grid):
    for D in SIZES:
        for k in (4.3, 17.7, 46.1):
            grid_res = D / k
            py, cc = make_pair(kind, D, dendrite_grid)
            X_py = rg.MonodisperseGenerator(py, NullRot(), grid_res).generate().T
            gen = _core.MonodisperseGenerator(cc, _core.NullRotator(), grid_res)
            X_cc = gen.lattice
            assert X_py.shape == X_cc.shape, (kind, D, grid_res)
            assert np.array_equal(X_py, X_cc), (kind, D, grid_res)


def test_lattice_fixed_resolution(dendrite_grid):
    # the measurement-device resolution used for the monomer dataset
    grid_res = 46.7e-6
    for kind in KINDS:
        for D in [100e-6, 333e-6, 1e-3, 2.5e-3]:
            py, cc = make_pair(kind, D, dendrite_grid)
            X_py = rg.MonodisperseGenerator(py, NullRot(), grid_res).generate().T
            X_cc = _core.MonodisperseGenerator(cc, _core.NullRotator(), grid_res).lattice
            assert np.array_equal(X_py, X_cc), (kind, D)


def test_is_inside_vectorized():
    cc = _core.Plate(1e-3)
    py = rc.Plate(1e-3)
    rng = np.random.default_rng(0)
    x, y, z = (rng.uniform(-6e-4, 6e-4, 5000) for _ in range(3))
    z *= 0.05
    assert np.array_equal(py.is_inside(x, y, z), cc.is_inside(x, y, z))

"""Aggregate kernels given identical inputs and random state.

Exact (bit-identical) comparisons: add_particle, add_elements/recentering,
remove_elements, projections, grid(), add_rime_particles, compact_rime,
STL export. Tolerance comparisons: operations that go through BLAS/LAPACK
in numpy (principal axes, align, minimum covering sphere).
"""
import os

import numpy as np
import pytest
from aggregation import aggregate as ra
from aggregation import mcs as rmcs
from aggregation import stl as rstl

from conftest import ArrayGen, random_aggregate_X
from snowagg import _core


def ref_agg(X, grid_res, cls=ra.Aggregate):
    return cls(ArrayGen(X, grid_res))


def synced_rng(seed):
    np.random.seed(seed)
    r = _core.Rng()
    r.set_state(np.random.get_state())
    return r


def same_state(r):
    st_np = np.random.get_state()
    st_cc = r.get_state()
    return np.array_equal(st_np[1], st_cc[1]) and st_np[2] == st_cc[2] and st_np[3] == st_cc[3]


@pytest.fixture(scope="module")
def aggs(dendrite_grid):
    out = []
    for seed, kind, n in [(1, "dendrite", 3), (2, "column", 4), (3, "plate", 2), (4, "dendrite", 5)]:
        out.append(random_aggregate_X(seed, n_mono=n, kind=kind, dendrite_grid=dendrite_grid).X.copy())
    return out


def test_add_elements_recentering(aggs):
    g = 20e-6
    for X in aggs:
        py = ref_agg(X, g)
        cc = _core.Aggregate(X, g)
        add = X[:37] + 1e-4
        py.add_elements(add, ident=3)
        cc.add_elements(add, 3)
        assert np.array_equal(py.X, cc.X)
        assert np.array_equal(py.ident, cc.ident)
        assert py.extent == cc.extent


def test_remove_elements(aggs):
    g = 20e-6
    X = aggs[0]
    py, cc = ref_agg(X, g), _core.Aggregate(X, g)
    rem = X[::7]
    py.remove_elements(rem)
    cc.remove_elements(rem)
    assert np.array_equal(py.X, cc.X)


@pytest.mark.parametrize("required", [True, False])
def test_add_particle_exact(aggs, required):
    g = 20e-6
    n_found = 0
    for trial in range(60):
        X = aggs[trial % len(aggs)]
        P = aggs[(trial + 1) % len(aggs)][:: (trial % 3) + 1] * (0.5 + 0.1 * (trial % 5))
        pen = [0.0, 10e-6, 80e-6][trial % 3]
        py = ref_agg(X, g)
        cc = _core.Aggregate(X, g)
        ident = np.arange(len(P), dtype=np.int32) % 5 + 1
        r = synced_rng(1000 + trial)
        ok_py = py.add_particle(particle=P, ident=ident, required=required, pen_depth=pen,
                                add_N_monomers=1, add_id_branch=1)
        ok_cc = cc.add_particle(P, ident, required, pen, r)
        assert ok_py == ok_cc
        n_found += ok_py
        assert same_state(r)
        assert np.array_equal(py.X, cc.X), trial
        assert np.array_equal(py.ident, cc.ident)
        assert py.extent == cc.extent
    assert n_found > 10


def test_projections(aggs):
    g = 20e-6
    for X in aggs:
        py, cc = ref_agg(X, g), _core.Aggregate(X, g)
        for dim in (0, 1, 2):
            assert np.array_equal(py.project_on_dim(dim=dim), cc.project_on_dim(dim))
            assert py.projected_area(dim=dim) == cc.projected_area(dim)
            assert py.projected_aspect_ratio(dim=dim) == _core.Aggregate.projected_aspect_ratio_of(
                cc.project_on_dim(dim))
        # arbitrary direction: rotation goes through BLAS in numpy
        p_py = py.project_on_dim(direction=(0.3, 1.1))
        p_cc = cc.project_on_direction(0.3, 1.1)
        assert abs(int(p_py.sum()) - int(p_cc.sum())) <= 0.01 * p_py.sum()


def test_principal_axes_and_align(aggs):
    g = 20e-6
    for X in aggs:
        py, cc = ref_agg(X, g), _core.Aggregate(X, g)
        pa_py, pa_cc = py.principal_axes(), cc.principal_axes()
        # eigenvector signs are arbitrary
        for c in range(3):
            s = np.sign(pa_py[:, c] @ pa_cc[:, c])
            assert np.allclose(pa_py[:, c], s * pa_cc[:, c], rtol=1e-9, atol=1e-18)
        assert np.isclose(py.aspect_ratio(), cc.aspect_ratio(), rtol=1e-12)
        py.align()
        cc.align()
        # aligned coordinates agree up to the sign of each axis
        for c in range(3):
            s = np.sign(py.X[:, c] @ cc.X[:, c])
            assert np.allclose(py.X[:, c], s * cc.X[:, c], rtol=0, atol=1e-12 * np.abs(X).max())


def test_grid_exact(aggs):
    for X in aggs:
        for res_mul in (1.0, 1.7, 2.9):  # coarse grids force many relocations
            g = 20e-6
            py, cc = ref_agg(X, g), _core.Aggregate(X, g)
            r = synced_rng(5)
            G_py = py.grid(res=g * res_mul)
            G_cc = cc.grid(g * res_mul, r)
            assert np.array_equal(G_py, G_cc)
            assert same_state(r)


def test_stl_identical(aggs, tmp_path):
    g = 20e-6
    X = aggs[2][:400]
    G = ref_agg(X, g).grid()
    f_py, f_cc = tmp_path / "py.stl", tmp_path / "cc.stl"
    rstl.snowflake_grid_to_stl(G, str(f_py))
    _core.snowflake_grid_to_stl(G, str(f_cc))
    assert f_py.read_bytes() == f_cc.read_bytes()
    assert os.path.getsize(f_py) > 1000


@pytest.mark.parametrize("compact_dist", [0.0, 0.5])
@pytest.mark.parametrize("N", [1, 2, 150])
def test_add_rime_particles_exact(aggs, StableRimedAggregate, N, compact_dist):
    g = 20e-6
    for i, X in enumerate(aggs):
        py = ref_agg(X, g, StableRimedAggregate)
        cc = _core.Aggregate(X, g)
        r = synced_rng(300 + i)
        py.add_rime_particles(N=N, pen_depth=120e-6, compact_dist=compact_dist)
        cc.add_rime_particles(N, 120e-6, compact_dist, r)
        assert same_state(r)
        assert np.array_equal(py.X, cc.X), (i, N, compact_dist)
        assert np.array_equal(py.ident, cc.ident)


def test_compact_rime_exact(aggs, StableRimedAggregate):
    g = 20e-6
    X = aggs[0]
    py = ref_agg(X, g, StableRimedAggregate)
    cc = _core.Aggregate(X, g)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = X[rng.integers(len(X))] + rng.normal(size=3) * g
        near = X[((X - p) ** 2).sum(1) < (2 * g) ** 2]
        for md in (0.3, 1.0, 5.0):
            a = py.compact_rime(p.copy(), near, max_dist=md)
            b = cc.compact_rime(list(p), near, md)
            assert np.array_equal(a, b)


def test_mcs_close(aggs):
    rng = np.random.default_rng(1)
    clouds = [rng.normal(size=(n, 3)) for n in (5, 50, 2000)] + [rng.uniform(size=(3000, 3))] + list(aggs)
    for P in clouds:
        c_py, r_py = rmcs.minimum_covering_sphere(P)
        c_cc, r_cc = _core.minimum_covering_sphere(P)
        scale = np.abs(P).max()
        assert np.isclose(r_py, r_cc, rtol=1e-12)
        assert np.allclose(c_py, c_cc, rtol=0, atol=1e-12 * scale)
        # it really is a covering sphere
        assert np.sqrt(((P - c_cc) ** 2).sum(1)).max() <= r_cc * (1 + 1e-12)


@pytest.mark.parametrize("n", [5, 1001, 8192, 8193, 100003, 1_000_003])
@pytest.mark.parametrize("order", ["C", "F"])
def test_recentering_matches_numpy_mean(n, order):
    """X -= X.mean(0): numpy's summation order depends on the memory layout
    (sequential for C order, chunked pairwise for Fortran order)."""
    X = np.random.default_rng(n).normal(size=(n, 3)) * 1e-3 + 2e-4
    X = np.asfortranarray(X) if order == "F" else np.ascontiguousarray(X)
    cc = _core.Aggregate(np.ascontiguousarray(X), 1e-5, 0, order == "F")
    cc.update_coordinates()
    X -= X.mean(0)
    assert np.array_equal(cc.X, X)

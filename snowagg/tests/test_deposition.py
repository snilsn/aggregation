"""Deposition growth / sublimation (deposition.grow_ice)."""
import numpy as np
import pytest
from aggregation import deposition as rdep

from conftest import random_aggregate_X
from test_aggregate_kernels import ref_agg, same_state, synced_rng
from snowagg import _core


@pytest.mark.parametrize("ice_vol_mul", [300, -150])
def test_grow_ice(dendrite_grid, ice_vol_mul):
    g = 20e-6
    n_exact = 0
    for seed in range(4):
        X = random_aggregate_X(seed + 10, n_mono=2, kind="dendrite", dendrite_grid=dendrite_grid).X.copy()
        py, cc = ref_agg(X, g), _core.Aggregate(X, g)
        r = synced_rng(seed)
        rdep.grow_ice(py, ice_vol_mul * g**3)
        _core.grow_ice(cc, r, ice_vol_mul * g**3)
        assert len(py.X) == len(cc.X) == len(X) + (ice_vol_mul if ice_vol_mul > 0 else ice_vol_mul)
        # The random walk only uses exact arithmetic, but the covering sphere
        # (numpy: BLAS/LAPACK) can differ in the last bits; report how often
        # the whole result is bit-identical and require closeness otherwise.
        if np.array_equal(py.X, cc.X) and same_state(r):
            n_exact += 1
        else:
            assert abs(py.X.mean(0) - cc.X.mean(0)).max() < 0.5 * g
    print("bit-identical deposition runs:", n_exact, "of 4")


@pytest.mark.parametrize("ice_vol_mul", [300, -150])
def test_grow_ice_exact_given_same_sphere(dendrite_grid, ice_vol_mul, monkeypatch):
    """With the covering sphere taken from the same implementation, the random
    walk, attachment and removal are bit-identical."""
    monkeypatch.setattr(rdep.mcs, "minimum_covering_sphere",
                        lambda X: _core.minimum_covering_sphere(np.ascontiguousarray(X)))
    g = 20e-6
    for seed in range(4):
        X = random_aggregate_X(seed + 10, n_mono=2, kind="dendrite", dendrite_grid=dendrite_grid).X.copy()
        py, cc = ref_agg(X, g), _core.Aggregate(X, g)
        r = synced_rng(seed)
        rdep.grow_ice(py, ice_vol_mul * g**3)
        _core.grow_ice(cc, r, ice_vol_mul * g**3, acos=np.arccos)
        assert same_state(r)
        assert np.array_equal(py.X, cc.X), seed
        assert np.array_equal(py.ident, cc.ident)

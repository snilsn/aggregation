"""Test setup: makes the Python reference package `aggregation` and the
development build of `snowagg` importable, and provides helpers."""
import inspect
import os
import pickle
import sys
import textwrap

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "aggregation"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, REF_DIR)

from aggregation import aggregate as ref_aggregate  # noqa: E402


class NullRot:
    """Identity rotator for the Python reference (its base Rotator.rotate
    has a broken signature)."""

    def rotate(self, X):
        return X


class ArrayGen:
    """Minimal Python-reference generator returning a fixed (N,3) array."""

    def __init__(self, X, grid_res):
        self.X = np.asarray(X, dtype=float)
        self.grid_res = grid_res

    def generate(self):
        return self.X.T.copy()


@pytest.fixture(scope="session")
def dendrite_grid():
    with open(os.path.join(REF_DIR, "aggregation", "dendrite_grid.dat"), "rb") as f:
        return pickle.load(f, encoding="latin1")


def _stable_sort_variant(cls, names):
    """Copy of methods of a reference class with argsort() made stable.

    numpy's default argsort is not stable (and on AVX-512 machines the order
    of ties is implementation specific), so the reference is made
    deterministic for exact comparisons. The C++ port uses a stable sort.
    """
    ns = dict(vars(ref_aggregate))
    body = {}
    for name in names:
        src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        src = src.replace(".argsort()", ".argsort(kind='stable')")
        exec(compile(src, f"<stable {name}>", "exec"), ns)
        body[name] = ns[name]
    return type(cls.__name__ + "Stable", (cls,), body)


@pytest.fixture(scope="session")
def StableRimedAggregate():
    return _stable_sort_variant(ref_aggregate.RimedAggregate, ["add_rime_particles", "compact_rime"])


def random_aggregate_X(seed, n_mono=3, D=0.4e-3, grid_res=20e-6, kind="dendrite", dendrite_grid=None):
    """A small but realistic aggregate built with the Python reference."""
    from aggregation import crystal, generator, rotator

    np.random.seed(seed)
    if kind == "dendrite":
        cry = crystal.Dendrite(D, hex_grid=dendrite_grid)
    elif kind == "column":
        cry = crystal.Column(D)
    else:
        cry = crystal.Plate(D)
    gen = generator.MonodisperseGenerator(cry, rotator.UniformRotator(), grid_res)
    agg = ref_aggregate.Aggregate(gen)
    for _ in range(n_mono - 1):
        agg.add_particle(required=True, pen_depth=grid_res)
        agg.align()
    return agg

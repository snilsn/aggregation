"""A 'hybrid' snowagg configuration for verification.

It runs the snowagg driver and all C++ kernels, but performs the few steps
that numpy delegates to BLAS/LAPACK (monomer rotation, rotate(), align(),
principal axes) with the original numpy code on arrays in the memory layout
the reference would have. If the hybrid reproduces the Python reference bit
for bit, the C++ kernels and the driver port are exact; the C++ versions of
the BLAS-level steps are checked separately by tolerance tests.
"""
import types

import numpy as np
from aggregation import aggregate as ref_aggregate
from aggregation import rotator as ref_rotator

from snowagg import _core
from snowagg import aggregate as sa
from snowagg import crystal as sc
from snowagg import generator as sg


def layout_X(agg):
    X = agg._c.X
    return np.asfortranarray(X) if agg._c.fortran_order else X


def set_X(agg, X):
    agg._c.X = np.ascontiguousarray(X)
    agg._c.fortran_order = bool(X.flags.f_contiguous and not X.flags.c_contiguous)
    agg._c.update_extent()
    agg._changed()


class HybridMixin:
    def principal_axes(self):
        X = layout_X(self)
        cov = X.T.dot(X) / X.shape[0]
        cov += np.diag(np.full(3, self.grid_res**2 / 12.))
        (l, v) = np.linalg.eigh(cov)
        return (v * np.sqrt(l))[:, ::-1]

    def align(self):
        PA = self.principal_axes()
        PA /= np.sqrt((PA**2).sum(0))
        set_X(self, np.dot(layout_X(self), PA))

    def rotate(self, rotator):
        X = layout_X(self)
        X = X - X.mean(0)
        set_X(self, rotator.rotate(X.T).T)

    def aspect_ratio(self):
        pa = self.principal_axes()
        pa_len = np.sqrt((pa**2).sum(axis=0))
        width = np.sqrt(0.5 * (pa_len[0]**2 + pa_len[1]**2))
        return pa_len[2] / width


class HybridAggregate(HybridMixin, sa.Aggregate):
    pass


class HybridRimedAggregate(HybridMixin, sa.RimedAggregate):
    pass


class HybridGenerator(sg.MonodisperseGenerator):
    """C++ lattice, numpy rotation (reference rotator)."""

    def __init__(self, crystal, rot, grid_res):
        self.crystal = crystal
        self.rot = rot
        self.grid_res = grid_res
        self._lattice_cc = _core.MonodisperseGenerator(crystal, _core.NullRotator(), grid_res).lattice
        self._c = None

    def generate(self):
        return self.rot.rotate(np.ascontiguousarray(self._lattice_cc.T))

    def _core_aggregate(self, ident):
        X = self.generate().T
        return _core.Aggregate(np.ascontiguousarray(X), self.grid_res, ident, bool(X.flags.f_contiguous))


def hybrid_modules():
    """Module replacements for snowagg.riming (aggregate, generator, rotator)."""
    agg_mod = types.SimpleNamespace(Aggregate=HybridAggregate, RimedAggregate=HybridRimedAggregate)
    gen_mod = types.SimpleNamespace(MonodisperseGenerator=HybridGenerator)
    return {"aggregate": agg_mod, "generator": gen_mod, "rotator": ref_rotator, "crystal": sc}


def stable_reference_modules(StableRimedAggregate):
    """The reference riming module with a deterministic (stable) argsort."""
    agg_mod = types.SimpleNamespace(Aggregate=ref_aggregate.Aggregate, RimedAggregate=StableRimedAggregate)
    return {"aggregate": agg_mod}

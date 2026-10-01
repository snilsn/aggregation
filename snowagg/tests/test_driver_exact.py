"""Whole aggregation + riming runs: the snowagg driver with the C++ kernels
reproduces the Python reference bit for bit (hybrid configuration, see
hybrid.py)."""
import numpy as np
import pytest
from aggregation import riming as R

import hybrid
from snowagg import riming as S

CONFIGS = {
    "dendrite_norime": (dict(psd="monodisperse", size=0.4e-3, mono_type="dendrite", grid_res=20e-6, rimed=True),
                        dict(N=5, riming_lwp=0.0)),
    "column_subsequent": (dict(psd="exponential", size=0.5e-3, min_size=0.1e-3, max_size=1.5e-3,
                               mono_type="column", grid_res=20e-6, rimed=True),
                          dict(N=5, riming_lwp=0.05, riming_mode="subsequent", lwp_div=5)),
    "plate_simultaneous": (dict(psd="exponential", size=0.5e-3, min_size=0.1e-3, max_size=1.5e-3,
                                mono_type="plate", grid_res=20e-6, rimed=True),
                           dict(N=4, riming_lwp=0.05, riming_mode="simultaneous", lwp_div=3)),
    "rosette_unaligned": (dict(psd="monodisperse", size=0.6e-3, mono_type="rosette", grid_res=20e-6, rimed=True),
                          dict(N=3, riming_lwp=0.02, riming_mode="subsequent", lwp_div=4, align=False)),
    "needle_compact": (dict(psd="exponential", size=0.8e-3, min_size=0.2e-3, max_size=2e-3,
                            mono_type="needle", grid_res=15e-6, rimed=True),
                       dict(N=4, riming_lwp=0.03, riming_mode="subsequent", lwp_div=3, compact_dist=0.5)),
    "bullet_single": (dict(psd="monodisperse", size=0.5e-3, mono_type="bullet", grid_res=20e-6, rimed=True),
                      dict(N=1, riming_lwp=0.05, lwp_div=2)),
    "spheroid_unrimed_agg": (dict(psd="exponential", size=0.4e-3, min_size=0.1e-3, max_size=1e-3,
                                  mono_type="spheroid", grid_res=20e-6, rimed=False),
                             dict(N=6, riming_lwp=0.0, riming_mode="subsequent")),
}


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("seed", [0, 1])
def test_driver_bit_identical(name, seed, monkeypatch, StableRimedAggregate):
    mono, kw = CONFIGS[name]
    for attr, val in hybrid.stable_reference_modules(StableRimedAggregate).items():
        monkeypatch.setattr(R, attr, val)
    for attr, val in hybrid.hybrid_modules().items():
        monkeypatch.setattr(S, attr, val)

    stages_py = [[(a.X.copy(), a.ident.copy()) for a in aggs]
                 for aggs in R.generate_rimed_aggregate(R.gen_monomer(**mono), seed=seed, iter=True, **kw)]
    stages_cc = [[(a.X.copy(), a.ident.copy()) for a in aggs]
                 for aggs in S.generate_rimed_aggregate(S.gen_monomer(**mono), seed=seed, iter=True, **kw)]
    assert len(stages_py) == len(stages_cc)
    for k, (sp, sc) in enumerate(zip(stages_py, stages_cc)):
        assert len(sp) == len(sc)
        for (Xp, ip), (Xc, ic) in zip(sp, sc):
            assert np.array_equal(ip, ic), (name, seed, k)
            assert np.array_equal(Xp, Xc), (name, seed, k)

"""Production configuration (everything in C++) vs the Python reference:
ensembles of aggregates have the same statistics.

Individual runs diverge after a few steps because the C++ principal axes can
come out with different eigenvector signs than LAPACK (which mirrors the
aligned aggregate), so only distributions can be compared here.
"""
import numpy as np
import pytest
from aggregation import mcs as rmcs
from aggregation import riming as R
from scipy import stats

from snowagg import mcs as smcs
from snowagg import riming as S

M = 40  # ensemble members per implementation

CONFIGS = {
    "dendrite_rimed": (dict(psd="monodisperse", size=0.5e-3, mono_type="dendrite", grid_res=20e-6, rimed=True),
                       dict(N=5, riming_lwp=0.05, riming_mode="subsequent", lwp_div=5)),
    "column_exp": (dict(psd="exponential", size=0.4e-3, min_size=0.1e-3, max_size=1.2e-3,
                        mono_type="column", grid_res=20e-6, rimed=True),
                   dict(N=6, riming_lwp=0.02, riming_mode="simultaneous", lwp_div=2)),
}


def properties(agg, mcs_fn):
    X = agg.X
    return dict(
        n=X.shape[0],
        n_rime=int((agg.ident == -1).sum()),
        D_max=2 * mcs_fn(X)[1],
        area=agg.vertical_projected_area(),
        area_side=agg.projected_area(dim=0),
        ar=agg.aspect_ratio(),
        z_extent=agg.extent[2][1] - agg.extent[2][0],
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_ensemble_statistics(name):
    mono, kw = CONFIGS[name]
    py = [properties(R.generate_rimed_aggregate(R.gen_monomer(**mono), seed=s, **kw), rmcs.minimum_covering_sphere)
          for s in range(M)]
    cc = [properties(S.generate_rimed_aggregate(S.gen_monomer(**mono), seed=10_000 + s, **kw),
                     smcs.minimum_covering_sphere) for s in range(M)]
    for key in py[0]:
        a = np.array([p[key] for p in py], dtype=float)
        b = np.array([p[key] for p in cc], dtype=float)
        p = stats.ks_2samp(a, b).pvalue
        print(f"{name:15s} {key:10s} py {a.mean():.4g}±{a.std():.2g}  cc {b.mean():.4g}±{b.std():.2g}  KS p={p:.3f}")
        assert p > 0.001, (name, key)

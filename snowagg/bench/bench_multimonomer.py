"""Workload of aggregation/notebooks/multiple_monomer_test.py (columns +
dendrites, 10 um, LWP 0.1, subsequent riming, lwp_div=500).

usage: python bench_multimonomer.py {snowagg|aggregation} N [N ...]
"""
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "aggregation"))
impl = sys.argv[1]
if impl == "snowagg":
    from snowagg import riming, fallvelocity, mcs
else:
    from aggregation import riming, fallvelocity, mcs

size = 500e-6
min_size, max_size, separation_size, grid_res = 100.0e-6, 3000.0e-6, 1000.0e-6, 10.0e-6
columns = dict(psd="exponential", size=size, min_size=min_size, max_size=separation_size,
               mono_type="column", grid_res=grid_res, rimed=True)
dendrites = dict(psd="exponential", size=size, min_size=separation_size, max_size=max_size,
                 mono_type="dendrite", grid_res=grid_res, rimed=True)
frac1 = stats.expon.cdf(separation_size, scale=size) - stats.expon.cdf(min_size, scale=size)
frac2 = stats.expon.cdf(max_size, scale=size) - stats.expon.cdf(separation_size, scale=size)
polygen = riming.gen_polydisperse_monomer(monomers=[columns, dendrites],
                                          ratios=[frac1 / (frac1 + frac2), frac2 / (frac1 + frac2)])

for N in map(int, sys.argv[2:]):
    t0 = time.time()
    n_stages = 0
    for aggs in riming.generate_rimed_aggregate(polygen, N=N, align=True, riming_lwp=0.1,
                                                riming_mode="subsequent", lwp_div=500, iter=True, seed=1):
        agg = aggs[0]
        n_stages += 1
    t1 = time.time()
    v = fallvelocity.fall_velocity(agg, method="HW")
    D = 2 * mcs.minimum_covering_sphere(agg.X)[1]
    print(f"{impl} N={N}: {t1 - t0:.1f} s, {n_stages} stages, {agg.X.shape[0]} elements, "
          f"D_max={D * 1e3:.2f} mm, v_HW={v:.2f} m/s", flush=True)

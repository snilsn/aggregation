"""Micro benchmark of the riming kernel on a ~0.5M element aggregate."""
import sys, time
import numpy as np
sys.path.insert(0, "python")
from snowagg import riming, _core

np.random.seed(3)
gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                         mono_type="dendrite", grid_res=10e-6, rimed=True)
agg = riming.generate_rimed_aggregate(gen, N=20, riming_lwp=0.0, seed=3)
print("aggregate elements:", len(agg))
for N in (20000, 200000):
    a = _core.Aggregate(np.ascontiguousarray(agg.X), agg.grid_res)
    r = _core.Rng(1)
    t = time.time()
    a.add_rime_particles(N, 120e-6, 0.0, r)
    dt = time.time() - t
    print(f"N={N}: {dt:.3f} s  ({dt / N * 1e6:.2f} us per rime particle)")

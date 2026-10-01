"""Generate an ensemble of rimed aggregates in parallel processes.

Each process runs single-threaded (OMP_NUM_THREADS=1) and uses its own seed,
so the ensemble is reproducible. Writes one line of properties per member.

    python parallel_ensemble.py out.txt
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before importing snowagg

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from snowagg import fallvelocity, mcs, riming  # noqa: E402

MONO = dict(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
            mono_type="dendrite", grid_res=10e-6, rimed=True)
RUN = dict(N=20, align=True, riming_lwp=0.1, riming_mode="subsequent", lwp_div=10)
RHO_I = 917.6


def member(seed):
    agg = riming.generate_rimed_aggregate(riming.gen_monomer(**MONO), seed=seed, **RUN)
    mass = RHO_I * len(agg) * agg.grid_res**3
    D_max = 2 * mcs.minimum_covering_sphere(agg)[1]
    area = agg.vertical_projected_area()
    v = fallvelocity.fall_velocity(agg, method="HW")
    return seed, mass, D_max, area, v


if __name__ == "__main__":
    seeds = range(32)
    with mp.Pool(os.cpu_count()) as pool:
        rows = pool.map(member, seeds)
    np.savetxt(sys.argv[1] if len(sys.argv) > 1 else "ensemble.txt", np.array(rows),
               header="seed mass[kg] D_max[m] area[m2] v_HW[m/s]", fmt="%.6e")

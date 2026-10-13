"""Mass-size relation of a small ensemble.

Aggregates of 1 to 30 dendrites (20 um elements), each with its own seed, once
unrimed and once rimed with 0.2 kg/m^2 of liquid water. A power law
m = a D^b fitted to each set shows how riming makes snowflakes heavier at the
same size. For large ensembles, run the members in parallel processes
(parallel_ensemble.py).

    python mass_size.py [mass_size.png]
"""
import sys

import numpy as np

from snowagg import mcs, riming

N_MONOMERS = np.unique(np.round(np.geomspace(1, 30, 16)).astype(int))


def member(n, lwp, seed):
    gen = riming.gen_monomer(psd="exponential", size=0.6e-3, min_size=0.1e-3, max_size=2e-3,
                             mono_type="dendrite", grid_res=20e-6, rimed=True)
    agg = riming.generate_rimed_aggregate(gen, N=n, riming_lwp=lwp, riming_mode="subsequent",
                                          lwp_div=max(1, round(lwp / 0.01)), seed=seed)
    mass = riming.rho_i * agg.num_elements * agg.grid_res**3
    D_max = 2 * mcs.minimum_covering_sphere(agg)[1]
    return D_max, mass


def main(png=None):
    sets = {}
    for lwp in (0.0, 0.2):
        D, m = np.array([member(n, lwp, seed=100 + k) for k, n in enumerate(N_MONOMERS)]).T
        b, log_a = np.polyfit(np.log(D), np.log(m), 1)
        sets[lwp] = (D, m, np.exp(log_a), b)
        print(f"LWP {lwp:.1f} kg/m^2: m = {np.exp(log_a):.3g} D^{b:.2f} (SI units), "
              f"D_max {D.min() * 1e3:.1f}-{D.max() * 1e3:.1f} mm")
    if png:
        import matplotlib
        import matplotlib.ticker
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for (lwp, (D, m, a, b)), c in zip(sets.items(), ("#2a78d6", "#eb6834")):
            ax.loglog(D * 1e3, m * 1e6, "o", color=c, label=f"LWP {lwp:g} kg m⁻²: m = {a:.3g} D^{b:.2f}")
            DD = np.geomspace(D.min(), D.max(), 50)
            ax.loglog(DD * 1e3, a * DD**b * 1e6, "-", color=c, lw=1)
        ax.set_xticks([0.5, 1, 2, 5])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("maximum dimension D_max [mm]")
        ax.set_ylabel("mass [mg]")
        ax.legend(fontsize=8)
        ax.set_title("Aggregates of 1-30 dendrites", fontsize=10)
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

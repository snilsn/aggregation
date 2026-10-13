"""The same snowflake with more and more rime.

With riming_mode="subsequent" the crystals are aggregated first and the
aggregate is rimed afterwards, so with the same seed every run builds the same
aggregate and only the rime differs. riming_lwp is the liquid water path the
particle falls through [kg/m^2]: the mass of droplets it collects per area of
its cross section. Each riming step (lwp_div steps in all) adds the droplets
of its share and turns the particle; within a step the droplets all come from
one direction, so use enough steps (here steps of 10 g/m^2). Elements of
20 um keep this example fast.

    python riming.py [riming.png]
"""
import sys

import numpy as np

from snowagg import fallvelocity, mcs, riming

LWPS = [0.0, 0.1, 0.3, 1.0]  # [kg/m^2]


def view(ax, X, pixel, along, title=""):
    """Column density of the elements (log scale) seen along axis `along`
    (0: x, 1: y, 2: z), in pixels of `pixel` [m]."""
    h, v = [a for a in (0, 1, 2) if a != along]
    P = np.floor((X[:, [h, v]] - X[:, [h, v]].min(0)) / pixel).astype(int)
    img = np.zeros(P.max(0) + 1)
    np.add.at(img, tuple(P.T), 1)
    x0, y0 = X[:, [h, v]].min(0) * 1e3
    ax.imshow(np.log1p(img.T), origin="lower", cmap="Blues", interpolation="nearest",
              extent=[x0, x0 + img.shape[0] * pixel * 1e3, y0, y0 + img.shape[1] * pixel * 1e3])
    ax.set_title(title, fontsize=9)


def main(png=None):
    gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                             mono_type="dendrite", grid_res=20e-6, rimed=True)
    particles = []
    for lwp in LWPS:
        agg = riming.generate_rimed_aggregate(gen, N=10, riming_lwp=lwp, riming_mode="subsequent",
                                              lwp_div=max(1, round(lwp / 0.01)), seed=1)
        mass = riming.rho_i * agg.num_elements * agg.grid_res**3
        rime = np.mean(agg.ident == -1)  # rime elements have the identifier -1
        D_max = 2 * mcs.minimum_covering_sphere(agg)[1]
        v = fallvelocity.fall_velocity(agg, T=263.15, P=1000e2)
        print(f"LWP {lwp:4.1f} kg/m^2: mass {mass * 1e6:5.2f} mg ({rime:3.0%} rime), D_max {D_max * 1e3:4.2f} mm, "
              f"fall speed {v:.2f} m/s")
        agg.align()  # longest axis along x, shortest along z: all seen the same way
        particles.append((lwp, agg.X - agg.X.mean(0), mass, rime, v))
    if png:
        import matplotlib
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(LWPS), figsize=(13, 3.8))
        for ax, (lwp, X, mass, rime, v) in zip(axes, particles):
            view(ax, X, 40e-6, along=1, title=f"LWP {lwp:g} kg m⁻²: {mass * 1e6:.2f} mg, {rime:.0%} rime\n"
                                              f"falls at {v:.2f} m s⁻¹")
            ax.set_xlim(-2.7, 2.7)
            ax.set_ylim(-1.9, 1.9)
            ax.set_xlabel("mm")
        fig.suptitle("The same aggregate of 10 dendrites rimed with more and more liquid water (side view)")
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

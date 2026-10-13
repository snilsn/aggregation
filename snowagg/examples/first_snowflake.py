"""A first snowflake: an aggregate of ten dendrites, lightly rimed.

The monomers are dendrites with sizes from an exponential distribution
(mean 1 mm, between 0.1 and 3 mm), made of volume elements of 10 um. They
are stuck together one pair at a time, then the aggregate collects
0.1 kg/m^2 of liquid water as rime. The result is an Aggregate object: the
coordinates of its elements and what follows from them.

    python first_snowflake.py [first_snowflake.png]
"""
import sys

import numpy as np

from snowagg import fallvelocity, mcs, riming


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
    # 1. a generator of monomers: each call returns a new crystal
    gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                             mono_type="dendrite", grid_res=10e-6, rimed=True)
    # 2. aggregation of 10 monomers, then riming (the seed makes it reproducible)
    agg = riming.generate_rimed_aggregate(gen, N=10, riming_lwp=0.1, riming_mode="subsequent",
                                          lwp_div=10, seed=1)
    # 3. what the snowflake is made of
    X = agg.X  # (n, 3) coordinates of the element centres [m]
    ident = agg.ident  # which monomer each element came from; rime: -1
    print(f"{len(X):,} elements of {agg.grid_res * 1e6:g} um, "
          f"{len(np.unique(ident[ident >= 0]))} monomers, rime {np.mean(ident == -1):.0%} of the mass")
    # 4. its properties
    mass = riming.rho_i * len(X) * agg.grid_res**3  # [kg]
    D_max = 2 * mcs.minimum_covering_sphere(agg)[1]  # maximum dimension [m]
    area = agg.projected_area(dim=2)  # seen from above [m^2]
    v = fallvelocity.fall_velocity(agg, T=263.15, P=1000e2)  # Heymsfield and Westbrook (2010) [m/s]
    print(f"mass {mass * 1e6:.3f} mg, D_max {D_max * 1e3:.2f} mm, projected area {area * 1e6:.2f} mm^2, "
          f"fall speed {v:.2f} m/s")
    if png:
        import matplotlib
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
        view(axes[0], X, 2 * agg.grid_res, along=2, title="seen from above (along z)")
        view(axes[1], X, 2 * agg.grid_res, along=1, title="seen from the side (along y)")
        for ax in axes:
            ax.set_xlabel("mm")
        fig.suptitle(f"10 dendrites, LWP 0.1 kg m⁻²: {mass * 1e6:.2f} mg, D_max {D_max * 1e3:.1f} mm, "
                     f"{v:.2f} m s⁻¹")
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

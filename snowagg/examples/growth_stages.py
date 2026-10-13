"""Watching a snowflake grow.

With iter=True, generate_rimed_aggregate returns the stages of the run one by
one: first the list of monomers, then the list of particles after every
merge, until one particle is left. With riming_mode="simultaneous" every
particle collects its share of the rime after every merge, as in a cloud
where crystals aggregate and rime at the same time.

    python growth_stages.py [growth_stages.png]
"""
import sys

import numpy as np

from snowagg import riming


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
    gen = riming.gen_monomer(psd="exponential", size=0.8e-3, min_size=0.1e-3, max_size=2e-3,
                             mono_type="plate", grid_res=10e-6, rimed=True)
    stages = riming.generate_rimed_aggregate(gen, N=6, riming_lwp=0.2, riming_mode="simultaneous",
                                             lwp_div=10, seed=3, iter=True)
    largest = []
    for k, particles in enumerate(stages):
        big = max(particles, key=lambda p: p.num_elements)
        rime = np.mean(big.ident == -1)
        print(f"stage {k}: {len(particles)} particle(s); the largest has {big.num_elements:,} elements, "
              f"{rime:.0%} rime")
        largest.append((k, len(particles), big.X - big.X.mean(0), rime))
    if png:
        import matplotlib
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(largest), figsize=(2.3 * len(largest), 3.0))
        for ax, (k, n, X, rime) in zip(axes, largest):
            view(ax, X, 20e-6, along=2, title=f"stage {k}: {n} particle{'s' if n > 1 else ''}\n{rime:.0%} rime")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("6 plates aggregating and riming at the same time: the largest particle of each stage "
                     "(seen from above, frame 3.2 mm)", fontsize=10)
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

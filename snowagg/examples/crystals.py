"""The seven crystal types of the model.

A monomer is a crystal shape filled with the points of a cubic grid of volume
elements (here 10 um) and turned into a random orientation. align() turns it
so that its longest axis is along x and its shortest along z: the top view
looks along z, the side view along y.

    python crystals.py [crystals.png]

prints the number of elements and the mass of each crystal and, with a file
name, draws them.
"""
import sys

import numpy as np

from snowagg import riming

TYPES = ["plate", "column", "needle", "dendrite", "rosette", "bullet", "spheroid"]
SIZE = 1e-3  # [m]: the diameter, or the length of columns and needles
GRID_RES = 10e-6  # edge of a volume element [m]


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
    crystals = {}
    for kind in TYPES:
        gen = riming.gen_monomer(psd="monodisperse", size=SIZE, mono_type=kind, grid_res=GRID_RES)
        cry = gen()  # an Aggregate made of one crystal
        cry.align()
        mass = riming.rho_i * cry.num_elements * GRID_RES**3
        print(f"{kind:9s} {cry.num_elements:7,} elements, mass {mass * 1e9:6.1f} ug")
        crystals[kind] = cry.X - cry.X.mean(0)
    if png:
        import matplotlib
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(TYPES), figsize=(14, 4.6))
        for j, (kind, X) in enumerate(crystals.items()):
            view(axes[0, j], X, 2 * GRID_RES, along=2, title=kind)
            view(axes[1, j], X, 2 * GRID_RES, along=1)
            for ax in axes[:, j]:
                ax.set_xlim(-0.55, 0.55)
                ax.set_ylim(-0.55, 0.55)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel("top view")
        axes[1, 0].set_ylabel("side view")
        fig.suptitle(f"Crystals of {SIZE * 1e3:g} mm, {GRID_RES * 1e6:g} µm elements (frame: 1.1 mm)")
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

"""Growth by vapour deposition, and sublimation.

deposition.grow_ice adds (or, with a negative volume, removes) ice with a
random walk: molecules diffuse in from far away and stick where they first
touch the particle, so exposed tips grow fastest. Here a dendrite of 1 mm
grows by half and by all of its own volume, and loses half of it.

    python deposition.py [deposition.png]
"""
import copy
import sys

import numpy as np

from snowagg import deposition, riming

CHANGES = [0.0, 0.5, 1.0, -0.5]  # added ice volume, relative to the crystal's volume


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
    np.random.seed(2)
    gen = riming.gen_monomer(psd="monodisperse", size=1e-3, mono_type="dendrite", grid_res=10e-6)
    crystal = gen()
    crystal.align()
    volume = crystal.num_elements * crystal.grid_res**3  # ice volume [m^3]
    grown = []
    for change in CHANGES:
        agg = copy.deepcopy(crystal)
        if change:
            deposition.grow_ice(agg, change * volume)
        print(f"{change:+.1f} x the volume: {agg.num_elements:,} elements "
              f"(deposited: {np.mean(agg.ident == deposition.DEPOSITION_IDENT):.0%})")
        grown.append((change, agg.X - agg.X.mean(0)))
    if png:
        import matplotlib
        matplotlib.use("Agg")  # draw into a file
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(CHANGES), figsize=(12, 3.4))
        for ax, (change, X) in zip(axes, grown):
            title = "the crystal" if change == 0 else (f"{change:+g} × its volume by deposition" if change > 0
                                                      else f"{change:+g} × its volume by sublimation")
            view(ax, X, 20e-6, along=2, title=title)
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("A dendrite of 1 mm grown and sublimated by deposition.grow_ice (seen from above)")
        fig.tight_layout()
        fig.savefig(png, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

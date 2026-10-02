"""The model in one figure: one snowagg run from the monomers to the rimed
aggregate (img/overview.png). Ice elements in blue, rime in orange, oblique view.

usage: python docs/overview.py [--seed 4]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from snowagg import mcs, riming  # noqa: E402

G = 10e-6
ICE, RIME = to_rgb("#1c5cab"), to_rgb("#eb6834")


def view(X, elev=25.0, azim=30.0):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return X @ (Rx @ Rz).T


def principal_frame(X, ref=None):
    """X in its principal-axes frame (largest extent along x, second along
    z), with the signs fixed by the third moment, so the same particle
    looks the same in every panel. ref: elements that define the frame."""
    ref = X if ref is None else ref
    c = ref.mean(0)
    w, V = np.linalg.eigh(np.cov((ref - c).T))
    V = V[:, [2, 0, 1]]  # largest -> x, smallest -> y (depth), middle -> z
    V *= np.sign(np.sum(((ref - c) @ V) ** 3, axis=0))
    return (X - c) @ V


def render(parts, half, pixel):
    """RGB image of (X, is_rime) parts (already placed), viewed along y."""
    edges = np.arange(-half, half + pixel, pixel)
    img = np.ones((len(edges) - 1, len(edges) - 1, 3)) * np.array(to_rgb(fs.SURFACE))
    for colour, sel in ((ICE, False), (RIME, True)):
        n = np.zeros((len(edges) - 1, len(edges) - 1))
        for X, rime in parts:
            P = view(X)[rime == sel] if sel else view(X)[~rime]
            if len(P):
                n += np.histogram2d(P[:, 2], P[:, 0], bins=[edges, edges])[0]
        alpha = (1 - np.exp(-n / (6.0 if sel else 4.0)))[..., None] * (0.85 if sel else 1.0)
        img = img * (1 - alpha) + np.array(colour) * alpha
    return img[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    fs.apply()
    gen = riming.gen_monomer(psd="exponential", size=0.5e-3, min_size=0.2e-3, max_size=1.5e-3,
                             mono_type="dendrite", grid_res=G, rimed=True)
    N, lwp, lwp_div = 8, 0.06, 6

    def run(lwp):
        return [[(np.asarray(a.X), np.asarray(a.ident) == -1) for a in aggs]
                for aggs in riming.generate_rimed_aggregate(gen, N=N, riming_lwp=lwp, riming_mode="subsequent",
                                                            lwp_div=lwp_div, seed=args.seed, iter=True)]
    # The driver yields no stage between the last merge and the first rime
    # increment; the same seed without riming aggregates identically.
    dry, wet = run(0.0), run(lwp)
    big = lambda st: max(st, key=lambda p: len(p[0]))  # noqa: E731
    picks = [
        ("1  monomers", "ice crystals on a grid of\nvolume elements", dry[0][:4]),
        ("2  aggregation", "falling crystals collide\nand stick", [big(dry[2])]),
        ("", f"all {N} crystals merged", [big(dry[-1])]),
        ("3  riming", f"droplets freeze where they hit\n(LWP {lwp / lwp_div:.2g} kg m⁻²)", [big(wet[N - 1])]),
        ("", f"LWP {lwp:g} kg m⁻²", [big(wet[-1])]),
    ]
    # the monomers side by side
    mono = []
    x = 0.0
    for X, r in picks[0][2]:
        X = X - X.mean(0)
        w = np.ptp(view(X)[:, 0])
        mono.append((X + np.array([x + w / 2, 0, 0]) @ np.linalg.inv(np.eye(3)), r))
        x += w + 0.1e-3
    shift = np.array([x / 2, 0, 0])
    picks[0] = (picks[0][0], picks[0][1], [(X - shift, r) for X, r in mono])
    half = 0.55 * max(np.ptp(view(np.vstack([p[0] for p in parts])), axis=0)[[0, 2]].max()
                      for _, _, parts in picks)
    fig, axes = plt.subplots(1, len(picks), figsize=(2.3 * len(picks), 3.1))
    for ax, (title, sub, parts) in zip(axes, picks):
        if title != picks[0][0]:
            # frame of the ice elements, so rime does not turn the particle
            parts = [(principal_frame(X, X[~r]), r) for X, r in parts]
        ax.imshow(render(parts, half, 1.5 * G), interpolation="antialiased",
                  extent=[-half * 1e3, half * 1e3, -half * 1e3, half * 1e3])
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=fs.INK)
        X = np.vstack([p[0] for p in parts])
        rime = np.concatenate([p[1] for p in parts])
        mass = riming.rho_i * len(X) * G**3
        if len(parts) == 1:
            d = 2 * mcs.minimum_covering_sphere(X)[1] + G
            info = f"D_max {d * 1e3:.1f} mm, {mass * 1e9:.0f} µg" + (f"\n{rime.mean():.0%} rime" if rime.any() else "")
        else:
            info = f"{len(parts)} of {N} crystals"
        ax.text(0.5, -0.02, sub + "\n" + info, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=fs.INK_2, linespacing=1.3)
    x0 = -half * 1e3 * 0.9
    axes[0].plot([x0, x0 + 0.5], [-half * 1e3 * 0.92] * 2, color=fs.INK_2, lw=1.5, solid_capstyle="butt")
    axes[0].text(x0 + 0.25, -half * 1e3 * 0.86, "0.5 mm", ha="center", fontsize=7, color=fs.INK_2)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.03, top=0.9, bottom=0.25)
    fs.save(fig, "overview.png")


if __name__ == "__main__":
    main()

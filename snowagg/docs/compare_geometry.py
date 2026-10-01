"""Figures comparing the geometry of snowagg's snowflakes with the original.

1. img/monomers.png: monomer lattices of all crystal types (bit-identical).
2. img/exact_run.png: a complete aggregation + riming run in the hybrid
   configuration of tests/hybrid.py (bit-identical at every stage).
3. img/ensemble.png, img/mass_size.png, img/gallery.png: ensembles of complete
   runs in the normal configuration, where individual runs differ (see the
   README) but the statistics are the same.

usage (from the snowagg directory, original package at ../aggregation):
    python docs/compare_geometry.py [--members 300] [--workers 6]

The ensemble properties are cached in docs/data/ensemble.csv; pass
--recompute to run the ensembles again.
"""
import argparse
import contextlib
import csv
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REF_DIR = os.path.abspath(os.path.join(ROOT, "..", "aggregation"))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, REF_DIR)

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(HERE, "data", "ensemble.csv")
RHO_I = 916.7

KINDS = [("plate", 1.0e-3), ("column", 1.0e-3), ("needle", 1.0e-3), ("dendrite", 2.0e-3),
         ("rosette", 1.0e-3), ("bullet", 0.6e-3), ("spheroid", 0.6e-3)]

EXACT_RUN = (dict(psd="exponential", size=0.5e-3, min_size=0.2e-3, max_size=1.5e-3,
                  mono_type="dendrite", grid_res=20e-6, rimed=True),
             dict(N=6, riming_lwp=0.1, riming_mode="subsequent", lwp_div=4), 3)

CONFIGS = {
    "Rimed dendrite aggregates":
        (dict(psd="exponential", size=0.6e-3, min_size=0.2e-3, max_size=2e-3, mono_type="dendrite",
              grid_res=20e-6, rimed=True),
         dict(N=10, riming_lwp=0.1, riming_mode="subsequent", lwp_div=5)),
    "Column aggregates, simultaneous riming":
        (dict(psd="exponential", size=0.4e-3, min_size=0.1e-3, max_size=1.2e-3, mono_type="column",
              grid_res=20e-6, rimed=True),
         dict(N=8, riming_lwp=0.05, riming_mode="simultaneous", lwp_div=2)),
    "Unrimed plate aggregates":
        (dict(psd="exponential", size=0.5e-3, min_size=0.1e-3, max_size=1.5e-3, mono_type="plate",
              grid_res=20e-6, rimed=True),
         dict(N=15, riming_lwp=0.0)),
}
PROPS = [("D_max", "maximum dimension $D_{max}$ [mm]", 1e3, True),
         ("mass", "mass [mg]", 1e6, True),
         ("area_ratio", "area ratio (vertical projection)", 1, False),
         ("aspect", "aspect ratio (principal axes)", 1, False),
         ("rime_fraction", "rime mass fraction", 1, False)]
GALLERY = 5  # members per implementation shown in img/gallery.png
SEED_OFFSET = {"original": 0, "snowagg": 100_000}


def packages(impl):
    if impl == "original":
        from aggregation import crystal, generator, mcs, riming
    else:
        from snowagg import crystal, generator, mcs, riming
    return crystal, generator, mcs, riming


def identity_rotator(impl):
    if impl == "original":
        from conftest import NullRot  # the original's base Rotator cannot rotate
        return NullRot()
    from snowagg import rotator
    return rotator.Rotator()


# ---- rendering ---------------------------------------------------------------

def view_rotation(elev=30.0, azim=25.0):
    """Fixed oblique viewing direction for the monomer pictures."""
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def projection(X, pixel, half_width, axes=(0, 2)):
    """Column density image (elements per pixel) of X projected on `axes`,
    centred on the centroid."""
    c = X.mean(0)
    edges = np.arange(-half_width, half_width + pixel, pixel)
    H, _, _ = np.histogram2d(X[:, axes[1]] - c[axes[1]], X[:, axes[0]] - c[axes[0]], bins=[edges, edges])
    return H[::-1]


def show(ax, H, half_width):
    ax.imshow(np.log1p(H), cmap=fs.DENSITY, interpolation="nearest",
              extent=[-half_width * 1e3, half_width * 1e3, -half_width * 1e3, half_width * 1e3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, half_width, length_mm):
    x0 = -half_width * 1e3 * 0.92
    y0 = -half_width * 1e3 * 0.9
    ax.plot([x0, x0 + length_mm], [y0, y0], color=fs.INK_2, lw=1.5, solid_capstyle="butt")
    ax.text(x0 + length_mm / 2, y0 + half_width * 1e3 * 0.05, f"{length_mm:g} mm", ha="center", va="bottom",
            fontsize=7, color=fs.INK_2)


# ---- 1. monomers ---------------------------------------------------------------

def monomer_lattices(g=10e-6):
    with open(os.path.join(REF_DIR, "aggregation", "dendrite_grid.dat"), "rb") as f:
        grid = pickle.load(f, encoding="latin1")
    out = []
    for impl in ("original", "snowagg"):
        crystal, generator, _, _ = packages(impl)
        rot = identity_rotator(impl)
        lat = {}
        for kind, D in KINDS:
            cry = {"plate": lambda: crystal.Plate(D), "column": lambda: crystal.Column(D),
                   "needle": lambda: crystal.Needle(D), "dendrite": lambda: crystal.Dendrite(D, hex_grid=grid),
                   "rosette": lambda: crystal.Rosette(D), "bullet": lambda: crystal.Bullet(D),
                   "spheroid": lambda: crystal.Spheroid(D, 0.6)}[kind]()
            lat[kind] = np.asarray(generator.MonodisperseGenerator(cry, rot, g).generate()).T
        out.append(lat)
    return out


def fig_monomers():
    g = 10e-6
    py, cc = monomer_lattices(g)
    R = view_rotation()
    fig, axes = plt.subplots(2, len(KINDS), figsize=(1.6 * len(KINDS), 3.9))
    for j, (kind, D) in enumerate(KINDS):
        same = py[kind].shape == cc[kind].shape and np.array_equal(py[kind], cc[kind])
        hw = 0.6 * D
        for i, X in enumerate((py[kind], cc[kind])):
            show(axes[i, j], projection(X @ R.T, 1.5 * g, hw), hw)
        axes[0, j].set_title(f"{kind}\n$D$ = {D * 1e3:g} mm", fontsize=9)
        axes[1, j].text(0.5, -0.08, f"{len(py[kind]):,} elements\n" + ("identical" if same else "DIFFERENT"),
                        transform=axes[1, j].transAxes, ha="center", va="top", fontsize=8, color=fs.INK_2)
        print(f"monomer {kind}: {len(py[kind])} elements, identical={same}")
    axes[0, 0].set_ylabel("original\n(Python)", fontsize=9, color=fs.INK)
    axes[1, 0].set_ylabel("snowagg\n(C++)", fontsize=9, color=fs.INK)
    scale_bar(axes[1, 0], 0.6e-3, 0.5)
    fig.suptitle("Monomer lattices, 10 µm elements, oblique view: the element coordinates are bit-identical",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fs.save(fig, "monomers.png")


# ---- 2. one complete run, hybrid configuration --------------------------------

@contextlib.contextmanager
def patched(module, attrs):
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def exact_run():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import hybrid
    from aggregation import aggregate as ref_aggregate
    from aggregation import riming as R
    from conftest import _stable_sort_variant
    from snowagg import riming as S

    stable = _stable_sort_variant(ref_aggregate.RimedAggregate, ["add_rime_particles", "compact_rime"])
    mono, kw, seed = EXACT_RUN
    with patched(R, hybrid.stable_reference_modules(stable)):
        st_py = [[(a.X.copy(), a.ident.copy()) for a in aggs]
                 for aggs in R.generate_rimed_aggregate(R.gen_monomer(**mono), seed=seed, iter=True, **kw)]
    with patched(S, hybrid.hybrid_modules()):
        st_cc = [[(a.X.copy(), a.ident.copy()) for a in aggs]
                 for aggs in S.generate_rimed_aggregate(S.gen_monomer(**mono), seed=seed, iter=True, **kw)]
    same = len(st_py) == len(st_cc) and all(
        len(a) == len(b) and all(np.array_equal(x1, x2) and np.array_equal(i1, i2) for (x1, i1), (x2, i2) in zip(a, b))
        for a, b in zip(st_py, st_cc))
    return st_py, st_cc, same


def fig_exact_run():
    st_py, st_cc, same = exact_run()
    mono, kw, seed = EXACT_RUN
    N = kw["N"]
    counts = [len(s) for s in st_py]
    # the driver yields after every merge except the last one, then after every
    # rime increment, then the final (re-oriented) particle
    rimed = counts.index(1)
    picks = [(1, "after 1 merge"), (N - 2, f"after {N - 2} merges"),
             (rimed, "all merged,\nfirst rime increment"),
             (len(st_py) - 1, f"final, LWP {kw['riming_lwp']:g} kg m$^{{-2}}$")]
    g = mono["grid_res"]
    big = lambda stage: max(stage, key=lambda t: len(t[0]))  # noqa: E731
    hw = 0.55 * max(np.ptp(big(st_py[k])[0], axis=0).max() for k, _ in picks)
    fig, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.6))
    for j, (k, label) in enumerate(picks):
        for i, st in enumerate((st_py, st_cc)):
            X, ident = big(st[k])
            show(axes[i, j], projection(X, 2 * g, hw), hw)
        X, ident = big(st_py[k])
        rime = (ident == -1).mean()
        axes[0, j].set_title(f"stage {k}\n{label}", fontsize=8.5)
        axes[1, j].text(0.5, -0.06, f"{len(X):,} elements" + (f", {rime:.0%} rime" if rime else ""),
                        transform=axes[1, j].transAxes, ha="center", va="top", fontsize=8, color=fs.INK_2)
    axes[0, 0].set_ylabel("original\n(Python)", fontsize=9, color=fs.INK)
    axes[1, 0].set_ylabel("snowagg\n(C++)", fontsize=9, color=fs.INK)
    scale_bar(axes[1, 0], hw, 1.0)
    verdict = "bit-identical" if same else "NOT identical"
    fig.suptitle(f"One aggregation + riming run (dendrites, 20 µm elements, seed {seed}), side view. "
                 f"Hybrid configuration:\nall {len(st_py)} stages, every element coordinate and label {verdict}",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.84)
    fs.save(fig, "exact_run.png")
    print(f"exact run: {len(st_py)} stages, identical={same}")


# ---- 3. ensembles, normal configuration ----------------------------------------

def member(task):
    impl, name, k = task
    if impl == "snowagg":
        import snowagg
        snowagg.set_num_threads(1)
    _, _, mcs, riming = packages(impl)
    mono, kw = CONFIGS[name]
    agg = riming.generate_rimed_aggregate(riming.gen_monomer(**mono), seed=SEED_OFFSET[impl] + k, **kw)
    X, ident, g = agg.X, agg.ident, agg.grid_res
    D_max = 2 * mcs.minimum_covering_sphere(X)[1]
    area = agg.vertical_projected_area()
    props = dict(impl=impl, config=name, member=k, n=len(X), D_max=D_max, mass=RHO_I * len(X) * g**3,
                 area_ratio=area / (np.pi / 4 * D_max**2), aspect=agg.aspect_ratio(),
                 rime_fraction=float((ident == -1).mean()))
    proj = None
    if name == next(iter(CONFIGS)) and k < GALLERY:
        proj = (X - X.mean(0)).astype(np.float32)
    return props, proj


def ensembles(members, workers, recompute):
    projs = {}
    if os.path.exists(DATA) and not recompute:
        with open(DATA) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for key in ("n", "member"):
                r[key] = int(r[key])
            for key, *_ in PROPS:
                r[key] = float(r[key])
        need = [(impl, next(iter(CONFIGS)), k) for impl in ("original", "snowagg") for k in range(GALLERY)]
    else:
        rows = []
        need = [(impl, name, k) for name in CONFIGS for impl in ("original", "snowagg") for k in range(members)]
    with mp.Pool(workers) as pool:
        for props, proj in pool.imap_unordered(member, need, chunksize=4):
            if not os.path.exists(DATA) or recompute:
                rows.append(props)
            if proj is not None:
                projs[(props["impl"], props["member"])] = proj
    if not os.path.exists(DATA) or recompute:
        rows.sort(key=lambda r: (r["config"], r["impl"], r["member"]))
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        with open(DATA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", DATA)
    return rows, projs


def values(rows, name, impl, key):
    return np.array([r[key] for r in rows if r["config"] == name and r["impl"] == impl], float)


def fig_ensemble(rows):
    names = list(CONFIGS)
    fig, axes = plt.subplots(len(names), len(PROPS), figsize=(2.55 * len(PROPS), 2.3 * len(names)))
    for i, name in enumerate(names):
        for j, (key, label, scale, logx) in enumerate(PROPS):
            ax = axes[i, j]
            a, b = values(rows, name, "original", key), values(rows, name, "snowagg", key)
            if key == "rime_fraction" and a.max() == 0 and b.max() == 0:
                ax.text(0.5, 0.5, "no riming", transform=ax.transAxes, ha="center", va="center", color=fs.MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue
            for v, colour, impl in ((a, fs.ORIGINAL, "original (Python)"), (b, fs.SNOWAGG, "snowagg (C++)")):
                x = np.sort(v * scale)
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=colour, lw=2, label=impl)
            if logx:
                ax.set_xscale("log")
                fs.plain_log(ax.xaxis)
            p = stats.ks_2samp(a, b).pvalue
            ax.text(0.97, 0.05, f"KS p = {p:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=fs.INK_2)
            ax.set_ylim(0, 1.02)
            if j == 0:
                ax.set_ylabel("cumulative fraction")
            else:
                ax.set_yticklabels([])
            if i == len(names) - 1 or (key == "rime_fraction"):
                ax.set_xlabel(label)
        axes[i, 0].set_title(f"{name} ({len(values(rows, name, 'original', 'n'))} runs each)", loc="left",
                             fontsize=9.5, x=0.0, pad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Ensembles of complete runs, normal configuration: same distributions",
                 fontsize=10.5, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fs.save(fig, "ensemble.png")


def fig_mass_size(rows):
    names = list(CONFIGS)
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), sharey=False)
    for ax, name in zip(axes, names):
        for impl, colour, label in (("original", fs.ORIGINAL, "original (Python)"),
                                    ("snowagg", fs.SNOWAGG, "snowagg (C++)")):
            D = values(rows, name, impl, "D_max") * 1e3
            m = values(rows, name, impl, "mass") * 1e6
            ax.scatter(D, m, s=14, color=colour, alpha=0.55, edgecolors=fs.SURFACE, linewidths=0.5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        fs.plain_log(ax.xaxis)
        fs.plain_log(ax.yaxis)
        ax.set_xlabel("$D_{max}$ [mm]")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("mass [mg]")
    axes[0].legend(loc="upper left", markerscale=1.5)
    fig.suptitle("Mass and size of every ensemble member", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, "mass_size.png")


def fig_gallery(projs):
    name = next(iter(CONFIGS))
    g = CONFIGS[name][0]["grid_res"]
    hw = 0.52 * max(np.ptp(X, axis=0).max() for X in projs.values())
    n = min(GALLERY, *(sum(1 for key in projs if key[0] == impl) for impl in ("original", "snowagg")))
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.3), squeeze=False)
    for i, impl in enumerate(("original", "snowagg")):
        for k in range(n):
            show(axes[i, k], projection(projs[(impl, k)], 2 * g, hw), hw)
    axes[0, 0].set_ylabel("original\n(Python)", fontsize=9, color=fs.INK)
    axes[1, 0].set_ylabel("snowagg\n(C++)", fontsize=9, color=fs.INK)
    scale_bar(axes[1, 0], hw, 1.0)
    fig.suptitle(f"{name}, first {n} ensemble members of each implementation (different seeds), side view",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.9)
    fs.save(fig, "gallery.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    fs.apply()
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    fig_monomers()
    fig_exact_run()
    rows, projs = ensembles(args.members, args.workers, args.recompute)
    for name in CONFIGS:
        for key, *_ in PROPS:
            a, b = values(rows, name, "original", key), values(rows, name, "snowagg", key)
            if a.max() > 0:
                print(f"{name:40s} {key:13s} original {a.mean():.4g}  snowagg {b.mean():.4g}  "
                      f"KS p={stats.ks_2samp(a, b).pvalue:.3f}")
    fig_ensemble(rows)
    fig_mass_size(rows)
    fig_gallery(projs)


if __name__ == "__main__":
    main()

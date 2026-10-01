"""Figures of the run times in docs/data/benchmark.csv (docs/benchmark.py):
img/speed_cases.png and img/speed_scaling.png.

usage: python docs/plot_benchmark.py
"""
import csv
import os

import numpy as np

import figstyle as fs
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [("original", "", "original (Python)", fs.ORIGINAL),
            ("snowagg", "1", "snowagg, 1 thread", fs.SNOWAGG),
            ("snowagg", "8", "snowagg, 8 threads", fs.SNOWAGG_MT)]


def load(path=os.path.join(HERE, "data", "benchmark.csv")):
    with open(path) as f:
        header = f.readline().lstrip("# ").strip()
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seconds"] = float(r["seconds"])
        r["N"] = int(r["N"]) if r["N"] else None
    return header, rows


def seconds_label(t):
    if t >= 100:
        return f"{t:.0f} s"
    if t >= 1:
        return f"{t:.1f} s"
    return f"{t:.2f} s"


def fig_cases(header, rows):
    cases = list(dict.fromkeys(r["case"] for r in rows if r["kind"] == "case"))
    groups = [(c, lambda r, c=c: r["kind"] == "case" and r["case"] == c) for c in cases]
    for N in (10, 20):
        groups.append((f"multiple_monomer_test workload, N = {N}\n(columns + dendrites, 10 µm, LWP 0.1, lwp_div 500)",
                       lambda r, N=N: r["kind"] == "scaling" and r["N"] == N))
    fig, ax = plt.subplots(figsize=(8.2, 0.95 * len(groups) + 0.9))
    bar_h, gap = 0.24, 0.03
    yticks = []
    for gi, (label, sel) in enumerate(groups):
        y0 = gi
        yticks.append(y0)
        t_orig = next(r["seconds"] for r in rows if sel(r) and r["impl"] == "original")
        for si, (impl, threads, name, colour) in enumerate(SETTINGS):
            t = next(r["seconds"] for r in rows if sel(r) and r["impl"] == impl and r["threads"] == threads)
            y = y0 + (si - 1) * (bar_h + gap)
            ax.barh(y, t, height=bar_h, color=colour, label=name if gi == 0 else None)
            text = seconds_label(t) if impl == "original" else f"{seconds_label(t)}   {t_orig / t:.0f}× faster"
            ax.text(t * 1.08, y, text, va="center", ha="left", fontsize=8, color=fs.INK_2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([g[0] for g in groups], fontsize=8.5, color=fs.INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    fs.plain_log(ax.xaxis, subs=(1,))
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    ax.set_xlabel("run time [s] (log scale)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.3)
    ax.set_title(f"Run time of the same workloads, same seed\n{header}", loc="left", fontsize=9.5, pad=28)
    fig.tight_layout()
    fs.save(fig, "speed_cases.png")


def fig_scaling(header, rows):
    sc = [r for r in rows if r["kind"] == "scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for impl, threads, name, colour in SETTINGS:
        pts = sorted((r["N"], r["seconds"]) for r in sc if r["impl"] == impl and r["threads"] == threads)
        N, t = np.array(pts).T
        ax.plot(N, t, "-o", color=colour, ms=7, mec=fs.SURFACE, mew=1.5, label=name)
        ax.text(N[-1] * 1.08, t[-1], seconds_label(t[-1]), va="center", fontsize=8, color=fs.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fs.plain_log(ax.xaxis)
    fs.plain_log(ax.yaxis)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    ax.set_xlabel("number of monomers N")
    ax.set_ylabel("run time [s]")
    ax.set_title("Run time", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    orig = {r["N"]: r["seconds"] for r in sc if r["impl"] == "original"}
    for impl, threads, name, colour in SETTINGS[1:]:
        pts = sorted((r["N"], orig[r["N"]] / r["seconds"]) for r in sc
                     if r["impl"] == impl and r["threads"] == threads and r["N"] in orig)
        N, s = np.array(pts).T
        ax.plot(N, s, "-o", color=colour, ms=7, mec=fs.SURFACE, mew=1.5, label=name)
        ax.text(N[-1] * 1.08, s[-1], f"{s[-1]:.0f}×", va="center", fontsize=8, color=fs.INK_2)
    ax.set_xscale("log")
    fs.plain_log(ax.xaxis)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("number of monomers N")
    ax.set_ylabel("speed-up (original / snowagg)")
    ax.set_title("Speed-up", loc="left")
    ax.legend(loc="upper left")
    fig.suptitle("multiple_monomer_test workload (columns + dendrites, 10 µm elements, LWP 0.1, lwp_div 500)\n"
                 + header, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, "speed_scaling.png")


def main():
    fs.apply()
    header, rows = load()
    fig_cases(header, rows)
    fig_scaling(header, rows)


if __name__ == "__main__":
    main()

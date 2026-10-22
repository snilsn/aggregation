"""Run times of the original Python `aggregation` package and snowagg.

Every case runs with the same parameters and seed in both implementations;
snowagg is timed with 1 thread and with 8 OpenMP threads. The original runs
with numpy's default settings. Writes docs/data/benchmark.csv, which
docs/plot_benchmark.py turns into the figures.

usage (from the snowagg directory; the original package is at the repository root):
    python docs/benchmark.py [--quick] [--only original|snowagg-1|snowagg-8]

Takes about half an hour, mostly the original; --quick runs the small cases only
(a check). --only runs one setting and replaces its rows in the CSV (so the slow
original need not be repeated). A full run does each setting in a process of
its own. The shortest of 3 runs is kept for the snowagg cases with N <= 20 (the
time of the first runs of a process varies, see warm_up).
"""
import argparse
import csv
import os
import platform
import sys
import time

_ap = argparse.ArgumentParser()
_ap.add_argument("--quick", action="store_true")
_ap.add_argument("--only", choices=["original", "snowagg-1", "snowagg-8"])
_ap.add_argument("--out", default=None)
ARGS = _ap.parse_args()
if ARGS.only == "snowagg-1":
    os.environ["OMP_NUM_THREADS"] = "1"  # before the OpenMP runtime starts

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repository root: the original package

import aggregation  # noqa: E402
import aggregation.dendrite  # noqa: E402
import aggregation.riming  # noqa: E402
import snowagg  # noqa: E402
import snowagg.dendrite  # noqa: E402
import snowagg.riming  # noqa: E402

IMPLS = {"original": aggregation, "snowagg": snowagg}


def dendrite_monomers(pkg, grid_res=10e-6):
    return pkg.riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                                  mono_type="dendrite", grid_res=grid_res, rimed=True)


def multimonomer(pkg):
    """The monomer mix of aggregation/notebooks/multiple_monomer_test.py."""
    size, sep = 500e-6, 1000e-6
    columns = dict(psd="exponential", size=size, min_size=100e-6, max_size=sep,
                   mono_type="column", grid_res=10e-6, rimed=True)
    dendrites = dict(psd="exponential", size=size, min_size=sep, max_size=3000e-6,
                     mono_type="dendrite", grid_res=10e-6, rimed=True)
    f1 = stats.expon.cdf(sep, scale=size) - stats.expon.cdf(100e-6, scale=size)
    f2 = stats.expon.cdf(3000e-6, scale=size) - stats.expon.cdf(sep, scale=size)
    return pkg.riming.gen_polydisperse_monomer(monomers=[columns, dendrites],
                                               ratios=[f1 / (f1 + f2), f2 / (f1 + f2)])


# each case: (label, function(pkg) -> number of elements of the result)
def case_reiter(pkg):
    g = pkg.dendrite.generate_dendrite(1.0, 0.4, 0.001, grid_size=400, num_iter=20000)
    return int((np.asarray(g) >= 1).sum())


def case_aggregation(pkg):
    agg = pkg.riming.generate_rimed_aggregate(dendrite_monomers(pkg), N=20, riming_lwp=0.0, seed=1)
    return agg.X.shape[0]


def case_riming(pkg):
    agg = pkg.riming.generate_rimed_aggregate(dendrite_monomers(pkg), N=10, riming_lwp=0.1,
                                              riming_mode="subsequent", lwp_div=10, seed=1)
    return agg.X.shape[0]


def case_multimonomer(N):
    def run(pkg):
        agg = pkg.riming.generate_rimed_aggregate(multimonomer(pkg), N=N, align=True, riming_lwp=0.1,
                                                  riming_mode="subsequent", lwp_div=500, seed=1)
        return agg.X.shape[0]
    return run


CASES = [
    ("Reiter dendrite growth (400x400 grid)", case_reiter),
    ("Aggregation, 20 dendrites (10 µm)", case_aggregation),
    ("Aggregation + riming, 10 dendrites (10 µm, LWP 0.1, lwp_div 10)", case_riming),
]
SCALING_N = {"original": [2, 5, 10, 20], "snowagg": [2, 5, 10, 20, 50, 100]}
QUICK = {"cases": [CASES[0]], "N": {"original": [2], "snowagg": [2]}}


def warm_up(pkg):
    """A few seconds of work that is not timed: the first parallel runs of a
    process are slow (threads start, the cores speed up from idle), which a
    single timed run of a few seconds would measure."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 4.0:
        pkg.dendrite.generate_dendrite(1.0, 0.4, 0.001, grid_size=200, num_iter=3000)


def timed(fn, pkg, repeats=1):
    """The result and the shortest of `repeats` run times."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(pkg)
        best = min(best, time.perf_counter() - t0)
    return out, best


SETTINGS = {"original": ("original", None), "snowagg-1": ("snowagg", 1), "snowagg-8": ("snowagg", 8)}


def main():
    args = ARGS
    out = args.out or os.path.join(HERE, "data", "benchmark.csv")
    if args.only is None:  # each setting in a process of its own (the OpenMP environment differs)
        import subprocess
        for name in SETTINGS:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--only", name, "--out", out]
                           + (["--quick"] if args.quick else []), check=True)
        return
    cases = QUICK["cases"] if args.quick else CASES
    scaling = QUICK["N"] if args.quick else SCALING_N
    names = [args.only]
    rows = []

    def record(kind, label, impl, threads, N, n_el, dt):
        rows.append(dict(kind=kind, case=label, impl=impl, threads=threads or "", N=N, elements=n_el,
                         seconds=f"{dt:.4f}"))
        print(f"{label:60s} {impl:8s} threads={threads or '-':2} {dt:9.2f} s  {n_el} elements", flush=True)

    for name in names:
        impl, threads = SETTINGS[name]
        if threads:
            snowagg.set_num_threads(threads)
        pkg = IMPLS[impl]
        if impl == "snowagg":
            warm_up(pkg)
        reps = 3 if impl == "snowagg" else 1  # the original takes minutes per run
        for label, fn in cases:
            n_el, dt = timed(fn, pkg, reps)
            record("case", label, impl, threads, "", n_el, dt)
        for N in scaling[impl]:
            n_el, dt = timed(case_multimonomer(N), pkg, reps if N <= 20 else 1)
            record("scaling", "multiple_monomer_test workload", impl, threads, N, n_el, dt)

    if os.path.exists(out):  # keep the rows of the other settings
        with open(out, newline="") as f:
            f.readline()
            old = list(csv.DictReader(f))
        impl, threads = SETTINGS[args.only]
        keep = [r for r in old if not (r["impl"] == impl and r["threads"] == (str(threads) if threads else ""))]
        rows = keep + rows
        rows.sort(key=lambda r: (["original", "snowagg"].index(r["impl"]), int(r["threads"] or 0), r["kind"] != "case"))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(f"# cpu: {cpu_name()}; python {platform.python_version()}, numpy {np.__version__}; "
                f"{time.strftime('%Y-%m-%d')}; snowagg: best of 3 runs (one run for N > 20)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    main()

"""Run times of the original Python `aggregation` package and snowagg.

Every case runs with the same parameters and seed in both implementations;
snowagg is timed with 1 thread and with 8 OpenMP threads. The original runs
with numpy's default settings. Writes docs/data/benchmark.csv, which
docs/plot_benchmark.py turns into the figures.

usage (from the snowagg directory; the original package is at the repository root):
    python docs/benchmark.py [--quick]

Takes about half an hour; --quick runs the small cases only (a check).
"""
import argparse
import csv
import os
import platform
import sys
import time

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


def timed(fn, pkg):
    t0 = time.perf_counter()
    out = fn(pkg)
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "benchmark.csv"))
    args = ap.parse_args()
    cases = QUICK["cases"] if args.quick else CASES
    scaling = QUICK["N"] if args.quick else SCALING_N
    settings = [("original", None), ("snowagg", 1), ("snowagg", 8)]
    rows = []

    def record(kind, label, impl, threads, N, n_el, dt):
        rows.append(dict(kind=kind, case=label, impl=impl, threads=threads or "", N=N, elements=n_el,
                         seconds=f"{dt:.4f}"))
        print(f"{label:60s} {impl:8s} threads={threads or '-':2} {dt:9.2f} s  {n_el} elements", flush=True)

    for impl, threads in settings:
        if threads:
            snowagg.set_num_threads(threads)
        pkg = IMPLS[impl]
        for label, fn in cases:
            n_el, dt = timed(fn, pkg)
            record("case", label, impl, threads, "", n_el, dt)
        for N in scaling[impl]:
            n_el, dt = timed(case_multimonomer(N), pkg)
            record("scaling", "multiple_monomer_test workload", impl, threads, N, n_el, dt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(f"# cpu: {cpu_name()}; python {platform.python_version()}, numpy {np.__version__}; "
                f"{time.strftime('%Y-%m-%d')}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


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

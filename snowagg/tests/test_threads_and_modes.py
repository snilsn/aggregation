"""Results do not depend on the number of OpenMP threads; RNG modes."""
import os
import subprocess
import sys

import numpy as np

from snowagg import riming, set_rng_mode

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = r"""
import sys, numpy as np
sys.path.insert(0, %r)
from snowagg import riming
gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.1e-3, max_size=3e-3,
                         mono_type="dendrite", grid_res=10e-6, rimed=True)
agg = riming.generate_rimed_aggregate(gen, N=6, riming_lwp=0.05, riming_mode="subsequent",
                                      lwp_div=4, seed=3)
np.save(sys.argv[1], agg.X)
""" % os.path.join(HERE, "..", "python")


def test_thread_count_independent(tmp_path):
    out = {}
    for nt in ("1", "3", "16"):
        f = tmp_path / f"x{nt}.npy"
        env = dict(os.environ, OMP_NUM_THREADS=nt)
        subprocess.run([sys.executable, "-c", SCRIPT, str(f)], env=env, check=True)
        out[nt] = np.load(f)
    assert len(out["1"]) > 100000  # large enough for the parallel code paths
    assert np.array_equal(out["1"], out["3"])
    assert np.array_equal(out["1"], out["16"])


# rime particles with compaction (the 3D index) and the covering sphere on a
# particle with more than 50000 elements: the parallel builds of the spatial
# indices and the parallel reductions must not depend on the thread count
COMPACT_SCRIPT = r"""
import sys, numpy as np
sys.path.insert(0, %r)
from snowagg import _core, riming, mcs
gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.5e-3, max_size=1.5e-3,
                         mono_type="dendrite", grid_res=5e-6, rimed=True)
base = riming.generate_rimed_aggregate(gen, N=3, riming_lwp=0.0, seed=4)
X = np.ascontiguousarray(base.X)
out = {"n": np.array(len(X))}
for name, compact in (("plain", 0.0), ("compact", 0.5)):
    a = _core.Aggregate(X.copy(), 5e-6, 0, True)
    a.add_rime_particles(3000, 120e-6, compact, _core.Rng(7))
    out[name] = np.asarray(a.X)
c, r = mcs.minimum_covering_sphere(X)
out["mcs"] = np.append(c, r)
np.savez(sys.argv[1], **out)
""" % os.path.join(HERE, "..", "python")


def test_compaction_and_mcs_thread_count_independent(tmp_path):
    out = {}
    for nt in ("1", "3", "16"):
        f = tmp_path / f"c{nt}.npz"
        env = dict(os.environ, OMP_NUM_THREADS=nt)
        subprocess.run([sys.executable, "-c", COMPACT_SCRIPT, str(f)], env=env, check=True)
        out[nt] = np.load(f)
    assert int(out["1"]["n"]) > 50000
    for nt in ("3", "16"):
        for k in out["1"].files:
            assert np.array_equal(out["1"][k], out[nt][k]), (nt, k)


def test_numpy_mode_uses_np_seed():
    gen = riming.gen_monomer(psd="monodisperse", size=0.5e-3, mono_type="plate", grid_res=20e-6, rimed=True)
    a = riming.generate_rimed_aggregate(gen, N=3, riming_lwp=0.02, riming_mode="subsequent", lwp_div=2, seed=5)
    b = riming.generate_rimed_aggregate(gen, N=3, riming_lwp=0.02, riming_mode="subsequent", lwp_div=2, seed=5)
    assert np.array_equal(a.X, b.X)


def test_own_rng_mode_leaves_numpy_state_alone():
    try:
        gen = riming.gen_monomer(psd="monodisperse", size=0.5e-3, mono_type="plate", grid_res=20e-6)
        np.random.seed(1)
        a = gen()
        st = np.random.get_state()
        set_rng_mode("own", seed=42)
        a.rotate(riming.rotator.UniformRotator())
        assert np.array_equal(np.random.get_state()[1], st[1])
        set_rng_mode("own", seed=42)
        b = gen()
        set_rng_mode("own", seed=42)
        c = gen()
        assert np.array_equal(b.X, c.X)
    finally:
        set_rng_mode("numpy")


def test_pickle_roundtrip():
    import pickle
    gen = riming.gen_monomer(psd="monodisperse", size=0.5e-3, mono_type="dendrite", grid_res=20e-6, rimed=True)
    a = riming.generate_rimed_aggregate(gen, N=3, riming_lwp=0.02, riming_mode="subsequent", lwp_div=2, seed=2)
    b = pickle.loads(pickle.dumps(a))
    assert type(b) is type(a)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.ident, b.ident)
    assert a.extent == b.extent and a.monomer_number == b.monomer_number and a.id_tree == b.id_tree
    b.add_rime_particles(N=20)  # still usable
    assert len(b) == len(a) + 20


def test_foreign_generator():
    from snowagg import aggregate

    class Gen:
        grid_res = 1e-5

        def generate(self):
            return np.random.default_rng(0).normal(size=(3, 50)) * 1e-4

    a = aggregate.Aggregate(Gen())
    assert len(a) == 50

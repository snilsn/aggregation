"""snowagg: C++ implementation of the `aggregation` snowflake aggregation
and riming model (J. Leinonen; OPTIMICe-team fork), with the same Python API.

Replace `from aggregation import X` by `from snowagg import X` for the
modules aggregate, crystal, dendrite, deposition, fallvelocity, generator,
mcs, riming, riming_runs, rotator and stl.

Random numbers: by default snowagg shares numpy's global random state, so
np.random.seed() works as before (see snowagg.set_rng_mode).
Threads: see snowagg.set_num_threads (or OMP_NUM_THREADS).
"""
import os as _os

from . import _core  # noqa: F401
from ._rng import get_rng_mode, set_rng_mode  # noqa: F401


def set_num_threads(n):
    """Number of OpenMP threads for whole-aggregate operations. Results do
    not depend on it."""
    _core.set_num_threads(int(n))


def get_num_threads():
    return _core.get_num_threads()


# Default: half the logical CPUs (about the physical cores), at most 8. The
# work is memory bound, and using every logical CPU makes each parallel
# region wait for threads that share a core with other processes.
if "OMP_NUM_THREADS" not in _os.environ:
    set_num_threads(max(1, min(8, (_os.cpu_count() or 2) // 2)))

__version__ = "0.1.0"

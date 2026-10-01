"""Random number handling.

By default ("numpy" mode) the C++ core draws from numpy's global legacy
random state: before each C++ call that needs random numbers the state of
np.random is copied into the C++ generator and afterwards copied back. So
np.random.seed() controls snowagg exactly as it controls the Python
`aggregation` package, and Python code that mixes np.random calls with
snowagg calls sees one consistent random stream.

In "own" mode snowagg uses its own generator (seed it with set_rng_mode),
which avoids the (small) state copying cost and leaves np.random untouched.
"""
import numpy as np

from . import _core

_rng = _core.Rng()
_mode = "numpy"


def set_rng_mode(mode, seed=None):
    """Select "numpy" (share np.random's global state) or "own"."""
    global _mode
    if mode not in ("numpy", "own"):
        raise ValueError("mode must be 'numpy' or 'own'")
    _mode = mode
    if mode == "own":
        if seed is None:
            _rng.seed_random()
        else:
            _rng.seed(int(seed))


def get_rng_mode():
    return _mode


class _Synced:
    __slots__ = ()

    def __enter__(self):
        if _mode == "numpy":
            _rng.set_state(np.random.get_state())
        return _rng

    def __exit__(self, *exc):
        if _mode == "numpy":
            np.random.set_state(_rng.get_state())
        return False


synced = _Synced()

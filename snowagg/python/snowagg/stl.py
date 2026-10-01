"""STL export (drop-in replacement for aggregation.stl)."""
import numpy as np

from . import _core


def snowflake_grid_to_stl(X, stl_fn, mode='ascii'):
    """Produce an ASCII STL file from a grid model.

    Args:
        X: (N,3) integer grid coordinates, e.g. from Aggregate.grid().
        stl_fn: The output STL file.
        mode: Only 'ascii' is supported.
    """
    if mode != 'ascii':
        raise ValueError("only mode='ascii' is supported")
    _core.snowflake_grid_to_stl(np.ascontiguousarray(X, dtype=np.int64), str(stl_fn))

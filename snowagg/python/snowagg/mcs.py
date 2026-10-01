"""Minimum covering sphere (drop-in replacement for aggregation.mcs)."""
import numpy as np

from . import _core


def minimum_covering_sphere(points):
    """Minimum covering sphere.

    Based on http://www.mel.nist.gov/msidlibrary/doc/hopp95.pdf

    Args:
        points: (N,3) array of points, or an Aggregate.

    Returns:
        A tuple (center, radius).
    """
    from .aggregate import Aggregate
    if isinstance(points, Aggregate):
        return _core.minimum_covering_sphere_agg(points._c)
    return _core.minimum_covering_sphere(np.ascontiguousarray(points, dtype=np.float64))

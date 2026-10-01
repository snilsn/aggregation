"""Deposition growth / sublimation (drop-in replacement for aggregation.deposition)."""
from . import _core
from ._rng import synced

DEPOSITION_IDENT = -2


def grow_ice(agg, ice_vol, outer_rad_norm=2., move_norm=1.):
    """Deposition growth or sublimation on an aggregate.

    Simulates deposition growth with a simple Monte Carlo (random walk)
    scheme.

    Args:
        agg: The Aggregate object.
        ice_vol: The volume of ice to be added [m^3]. If negative,
            sublimation is simulated instead.
        outer_rad_norm: Limiting distance for the diffusion scheme as a
            multiple of the particle radius (default 2).
        move_norm: The size scale of the diffusion process as a multiple of
            the aggregate volume element size.
    """
    with synced as rng:
        _core.grow_ice(agg._c, rng, ice_vol, outer_rad_norm, move_norm)
    agg._changed()

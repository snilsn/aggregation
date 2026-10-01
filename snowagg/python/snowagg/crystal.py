"""Ice crystal geometries (drop-in replacement for aggregation.crystal).

The classes are the C++ implementations; is_inside() accepts numpy arrays.
"""
import numpy as np

from . import _core

# numpy's value of tan(28 deg), which on some CPUs differs from libm's in
# the last bit; used so that rosette/bullet dimensions equal the reference.
_TAN_ALPHA = float(np.tan(28 * (np.pi / 180.0)))

Crystal = _core.Crystal
Plate = _core.Plate
Column = _core.Column
Needle = _core.Needle
Spheroid = _core.Spheroid


class Rosette(_core.Rosette):
    """6-branch bullet rosette crystal geometry; D is the maximum diameter."""

    def __init__(self, D):
        super().__init__(D, _TAN_ALPHA)


class Bullet(_core.Bullet):
    """Bullet crystal geometry; D is the maximum diameter."""

    def __init__(self, D):
        super().__init__(D, _TAN_ALPHA)


class Dendrite(_core.Dendrite):
    """Dendrite crystal geometry.

    Constructor args:
        D: The diameter of the dendrite.
        hex_grid: Hexagonal grid with the pregenerated dendrite shape.
        alpha, beta, gamma, num_iter, grid_size: Parameters for generating
            the dendrite if hex_grid is None (see snowagg.dendrite).
    """

    def __init__(self, D, alpha=1.0, beta=0.35, gamma=0.001, num_iter=5000, grid_size=400, hex_grid=None):
        if hex_grid is None:
            from . import dendrite
            hex_grid = dendrite.generate_dendrite(alpha, beta, gamma, num_iter=num_iter, grid_size=grid_size)
        super().__init__(D, np.ascontiguousarray(hex_grid, dtype=float), grid_size)
        self.ice = hex_grid >= 1.0


def crystal_by_temperature(tem):
    """Simple classification of crystals by temperature.

    Args:
        tem: Temperature in Kelvin.
    """
    tem_C = tem - 273.15
    if tem_C > -3:
        return Dendrite
    elif tem_C > -10:
        return Needle
    elif tem_C > -22:
        return Dendrite
    else:
        return Column

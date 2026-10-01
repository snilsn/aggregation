"""Crystal generators (drop-in replacement for aggregation.generator)."""
import numpy as np

from . import _core
from ._rng import synced


class Generator(object):
    """Base class for all generators, which should implement generate()."""

    def generate(self):
        pass


class MonodisperseGenerator(Generator):
    """Crystal generator for monodisperse particles.

    Constructor args:
        crystal: A crystal from snowagg.crystal (a crystal written in Python
            with max_radius() and a vectorized is_inside(x,y,z) also works).
        rot: A rotator from snowagg.rotator.
        grid_res: The grid spacing of the volume elements on a Cartesian grid.
    """

    def __init__(self, crystal, rot, grid_res):
        self.crystal = crystal
        self.rot = rot
        self.grid_res = grid_res
        if isinstance(crystal, _core.Crystal) and isinstance(rot, _core.Rotator):
            self._c = _core.MonodisperseGenerator(crystal, rot, grid_res)
        else:
            self._c = None
            self._lattice = self._python_lattice()

    def _python_lattice(self):
        # the original numpy implementation, for crystals defined in Python
        grid_res = self.grid_res
        max_r = self.crystal.max_radius()
        max_r += grid_res - max_r % grid_res
        (x, y, z) = np.mgrid[-max_r:max_r + grid_res * 0.001:grid_res,
                             -max_r:max_r + grid_res * 0.001:grid_res,
                             -max_r:max_r + grid_res * 0.001:grid_res]
        inside = self.crystal.is_inside(x, y, z)
        return np.vstack((x[inside], y[inside], z[inside]))

    def generate(self):
        """Create a volume-element realization of the crystal.

        Returns:
            The (3,N) array of volume element coordinates.
        """
        if self._c is not None:
            with synced as rng:
                return self._c.generate(rng).T
        return self.rot.rotate(self._lattice.copy())

    def _core_aggregate(self, ident):
        if self._c is not None:
            with synced as rng:
                return _core.Aggregate(self._c, rng, ident)
        X = self.generate().T
        return _core.Aggregate(np.ascontiguousarray(X), self.grid_res, ident, True)

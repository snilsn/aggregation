"""Volume-element aggregate models (drop-in replacement for
aggregation.aggregate), backed by the C++ core.

Differences to the original API:
  * `X` and `ident` return read-only copies (assigning a new array to them
    works); `len(agg)` / `agg.num_elements` give the element count cheaply.
  * add_particle(particle=...) also accepts another Aggregate.
"""
import numpy as np

from . import _core, rotator
from ._rng import synced


def get_proj_area_from_alphashape(proj_grid, alpha=0.4):
    """Projected area from an alpha shape of the projected grid (in pixels).

    Requires the alphashape and shapely packages.
    """
    import alphashape
    from shapely.geometry import Point

    coord = np.where(proj_grid > 0)
    alpha_shape = alphashape.alphashape(np.column_stack((coord[0], coord[1])), alpha)
    if proj_grid.shape[0] * proj_grid.shape[1] > 10000:
        area = alpha_shape.area
    else:
        area = 0
        for i in range(0, proj_grid.shape[0]):
            for j in range(0, proj_grid.shape[1]):
                if Point(i, j).intersects(alpha_shape):
                    area += 1
    return area


class Aggregate(object):
    """A volume-element aggregate model.

    Constructor args:
        generator: The crystal generator used to make this aggregate.

    Constructor keyword args:
        ident: The numerical identifier for this particle (integer, default 0).
    """

    def __init__(self, generator, ident=0):
        self._generator = generator
        self.grid_res = generator.grid_res
        if hasattr(generator, "_core_aggregate"):
            self._c = generator._core_aggregate(ident)
        else:  # any object with generate() -> (3,N) array and grid_res
            X = np.asarray(generator.generate(), dtype=float).T
            self._c = _core.Aggregate(np.ascontiguousarray(X), self.grid_res, ident, True)
        self._X_cache = None
        self.monomer_number = 1
        self.id_tree = ident

    # pickling (e.g. to return aggregates from multiprocessing workers); the
    # generator is not pickled, so add_particle() needs an explicit particle
    def __getstate__(self):
        st = self.__dict__.copy()
        c = st.pop("_c")
        st["_core_state"] = (c.X, c.ident, c.extent, c.fortran_order)
        st["_X_cache"] = None
        st["_generator"] = None
        return st

    def __setstate__(self, st):
        X, ident, extent, fortran_order = st.pop("_core_state")
        self.__dict__.update(st)
        self._c = _core.Aggregate(np.ascontiguousarray(X), self.grid_res, 0, fortran_order)
        self._c.ident = ident
        self._c.extent = extent

    # ---- data access -------------------------------------------------------
    def _changed(self):
        self._X_cache = None

    @property
    def X(self):
        """(N,3) array of element coordinates (read-only copy)."""
        if self._X_cache is None:
            X = self._c.X
            X.flags.writeable = False
            self._X_cache = X
        return self._X_cache

    @X.setter
    def X(self, value):
        value = np.asarray(value, dtype=float)
        fortran = bool(value.flags.f_contiguous and not value.flags.c_contiguous)
        self._c.X = np.ascontiguousarray(value)
        self._c.fortran_order = fortran
        self._changed()

    @property
    def ident(self):
        return self._c.ident

    @ident.setter
    def ident(self, value):
        self._c.ident = value

    @property
    def extent(self):
        return self._c.extent

    @extent.setter
    def extent(self, value):
        self._c.extent = value

    @property
    def num_elements(self):
        return len(self._c)

    def __len__(self):
        return len(self._c)

    # ---- geometry ------------------------------------------------------------
    def update_extent(self):
        self._c.update_extent()

    def project_on_dim(self, dim=2, direction=None):
        """2D projection along dim (0, 1 or 2) or from direction=(alpha, beta)."""
        if direction is not None:
            (alpha, beta) = direction
            proj = self._c.project_on_direction(alpha, beta)
            self._changed()
            return proj
        return self._c.project_on_dim(dim)

    def projected_area(self, dim=2, direction=None, method="default"):
        if method == "alphashape":
            proj_grid = self.project_on_dim(dim=dim)
            return get_proj_area_from_alphashape(proj_grid, alpha=0.5) * self.grid_res**2
        if method != "default":
            print("method \"" + method + "\" not implemented in projected area; use default()")
        if direction is None:
            return self._c.projected_area(dim)
        return self.project_on_dim(dim=dim, direction=direction).sum() * self.grid_res**2

    def vertical_projected_area(self):
        # Deprecated, for backward compatibility
        return self.projected_area(dim=2)

    def projected_aspect_ratio(self, dim=2, direction=None):
        proj_grid = np.ascontiguousarray(self.project_on_dim(dim=dim, direction=direction))
        return _core.Aggregate.projected_aspect_ratio_of(proj_grid)

    def aspect_ratio(self):
        return self._c.aspect_ratio()

    def principal_axes(self):
        return self._c.principal_axes()

    def align(self):
        """Align the longest principal axis with x and the shortest with z."""
        self._c.align()
        self._changed()

    def rotate(self, rotator):
        """Rotate the aggregate (around its center) with the given rotator."""
        if isinstance(rotator, _core.Rotator):
            with synced as rng:
                self._c.rotate(rotator, rng)
        else:  # a rotator written in Python
            X = self.X - self.X.mean(0)
            self.X = rotator.rotate(X.T).T
            self._c.update_extent()
        self._changed()

    def grid(self, res=None):
        """Integer grid coordinates of the elements (see the original docs)."""
        if res is None:
            res = self.grid_res
        with synced as rng:
            return self._c.grid(res, rng)

    # ---- growth ----------------------------------------------------------------
    def add_particle(self, particle=None, ident=None, required=False, pen_depth=0.0,
                     add_N_monomers=None, add_id_branch=None):
        """Merge another particle into this one.

        The other particle is added at a random location in the (x,y) plane
        and at the bottom of this particle in the z direction.

        Args:
            particle: (N,3) array of the other particle's elements, or another
                Aggregate, or None to generate a new monomer.
            ident: identifiers of the added elements (default: those of the
                other Aggregate, or 0).
            required: keep trying until a merging point is found.
            pen_depth: distance the particle may penetrate into this one.

        Returns:
            True if the merge was successful, False otherwise.
        """
        if particle is None:
            particle = self._generator.generate().T
            add_N_monomers = 1
            add_id_branch = -9999
        if isinstance(particle, Aggregate) and ident is None:
            with synced as rng:
                site_found = self._c.add_aggregate(particle._c, required, pen_depth, rng)
        else:
            if isinstance(particle, Aggregate):
                particle = particle.X
            particle = np.ascontiguousarray(particle, dtype=float)
            if ident is None:
                ident = np.zeros(particle.shape[0], dtype=np.int32)
            with synced as rng:
                site_found = self._c.add_particle(particle, ident, required, pen_depth, rng)
        self._changed()
        if site_found:
            self.monomer_number += add_N_monomers
            self.id_tree = [self.id_tree, add_id_branch]
        return site_found

    def add_elements(self, added_elements, ident=0, update=True):
        added = np.asarray(added_elements, dtype=float)
        fortran = bool(added.flags.f_contiguous and not added.flags.c_contiguous)
        self._c.add_elements(np.ascontiguousarray(added), ident, update, fortran)
        self._changed()

    def remove_elements(self, removed_elements, tolerance=0.001, update=True):
        self._c.remove_elements(np.ascontiguousarray(removed_elements, dtype=float), tolerance, update)
        self._changed()

    def update_coordinates(self):
        """Recenter the aggregate and update the particle extent."""
        self._c.update_coordinates()
        self._changed()

    def visualize(self, bgcolor=(1, 1, 1), fgcolor=(.8, .8, .8)):
        """Visualize the aggregate using Mayavi."""
        from matplotlib import colors
        from mayavi import mlab

        color_list = [colors.colorConverter.to_rgb(c) for c in [
            "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
            "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
            "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
        ]]
        mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor)
        i = 0
        X_all, ident_all = self.X, self.ident
        for ident in range(ident_all.min(), ident_all.max() + 1):
            X = X_all[ident_all == ident, :]
            if X.shape[0] > 0:
                mlab.points3d(X[:, 0], X[:, 1], X[:, 2], color=color_list[i % len(color_list)],
                              mode="cube", scale_factor=self.grid_res)
                i += 1


class RimedAggregate(Aggregate):
    """A volume-element rimed aggregate model (adds add_rime_particles)."""

    RIME_IDENT = -1

    def add_rime_particles(self, N=1, pen_depth=120e-6, compact_dist=0.):
        """Add N rime particles falling vertically onto the aggregate.

        Args:
            N: Number of rime particles to add.
            pen_depth: The distance that the rime particle is allowed to
                penetrate inside this particle.
            compact_dist: Maximum distance (in grid_res) a rime particle is
                moved by the compaction step (0 disables compaction).
        """
        with synced as rng:
            self._c.add_rime_particles(int(N), pen_depth, compact_dist, rng)
        self._changed()

    def compact_rime(self, X, X_near, max_dist=0., min_move=0.01, dr=0.1, max_iters=100):
        return np.array(self._c.compact_rime(list(X), np.ascontiguousarray(X_near, dtype=float),
                                             max_dist, min_move, dr, max_iters))

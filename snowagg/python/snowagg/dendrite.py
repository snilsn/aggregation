"""Reiter (2005) dendrite growth (drop-in replacement for aggregation.dendrite)."""
from . import _core


def generate_dendrite(alpha, beta, gamma, grid_size=1000, num_iter=10000, verbose=False):
    """Generate a 2D dendrite on a hexagonal grid (Reiter 2005,
    https://doi.org/10.1016/j.chaos.2004.06.071).

    Args:
        alpha, beta, gamma: Parameters of the algorithm.
        grid_size: The number of elements per dimension in the 2D grid.
        num_iter: Maximum number of iterations.
        verbose: If True, also return the number of iterations and the
            margin to the grid boundary.
    """
    (grid, it, margin) = _core.generate_dendrite(alpha, beta, gamma, grid_size, num_iter)
    if verbose:
        print("Reiter algorithm stopped at {} iterations with margin {}".format(it, margin))
        return grid, it, margin
    return grid

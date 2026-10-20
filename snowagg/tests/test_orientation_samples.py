"""orientation_samples (the loop of get_N_rime_particles: area, align, rotate,
with the sums for the principal axes made while rotating) gives the same areas
and the same particle as calling the three steps one by one; and the projected
area does not depend on the extent being exact."""
import numpy as np
import pytest

from snowagg import riming, rotator


def particle(seed=3):
    gen = riming.gen_monomer(psd="exponential", size=1e-3, min_size=0.5e-3, max_size=1.5e-3,
                             mono_type="dendrite", grid_res=5e-6, rimed=True)
    np.random.seed(seed)
    agg = riming.generate_rimed_aggregate(gen, N=3, riming_lwp=0.0, seed=seed)
    assert len(agg) > 100000  # more than the parallel threshold, several blocks of rows
    return agg


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("rot", [rotator.UniformRotator(), rotator.HorizontalRotator(), rotator.NullRotator()])
def test_same_as_stepwise(align, rot):
    a, b = particle(), particle()
    np.random.seed(11)
    areas = a.orientation_samples(rot, 4, align)
    state_a = np.random.get_state()[1]
    np.random.seed(11)
    ref = []
    for _ in range(4):
        ref.append(b.vertical_projected_area())
        if align:
            b.align()
        b.rotate(rot)
    assert areas == ref
    assert np.array_equal(a.X, b.X)
    assert a.extent == b.extent
    assert np.array_equal(state_a, np.random.get_state()[1])  # the same random numbers were drawn


def test_python_rotator_falls_back():
    class Spin:  # a rotator written in Python: rotates the (3, N) coordinates about z by 90 degrees
        def rotate(self, X):
            return np.array([-X[1], X[0], X[2]])

    a, b = particle(), particle()
    areas = a.orientation_samples(Spin(), 2, True)
    ref = []
    for _ in range(2):
        ref.append(b.vertical_projected_area())
        b.align()
        b.rotate(Spin())
    assert areas == ref


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_projected_area_with_wrong_extent(scale):
    a = particle()
    ext = np.array(a.extent, dtype=float)
    a.extent = (ext * [[1.0, scale]] * 1).tolist() if scale != 1.0 else ext.tolist()  # upper edge too small or large
    for dim in range(3):
        grid = a.project_on_dim(dim)
        assert a.projected_area(dim) == grid.sum() * a.grid_res ** 2

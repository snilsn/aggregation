"""Random rotations (drop-in replacement for aggregation.rotator).

rotate(X) takes and returns a (3,N) array like the original; aggregates
use the C++ implementation directly.
"""
import numpy as np
from numpy import array, pi

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:  # old scipy
    from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from . import _core
from ._rng import synced


def _rotation_matrix(alpha, beta, psi):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cp = np.cos(psi)
    sp = np.sin(psi)
    Rb = array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    Ra = array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rp = array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    return np.dot(np.dot(Ra, Rb), Rp)


class _RotateMixin:
    rotation_matrix = staticmethod(_rotation_matrix)

    def rotate(self, X):
        """Rotate the (3,N) coordinates around the [0,0,0] point."""
        X = np.asarray(X, dtype=float)
        with synced as rng:
            return _core.Rotator.rotate(self, np.ascontiguousarray(X.T), rng).T


class Rotator(_RotateMixin, _core.NullRotator):
    """Identity rotation (base class). Unlike the original, rotate() works."""

    def __init__(self):
        _core.NullRotator.__init__(self)


NullRotator = Rotator


class UniformRotator(_RotateMixin, _core.UniformRotator):
    """Uniformly random rotation."""

    def __init__(self):
        _core.UniformRotator.__init__(self)


class HorizontalRotator(_RotateMixin, _core.HorizontalRotator):
    """Uniformly random rotation around the z axis only."""

    def __init__(self):
        _core.HorizontalRotator.__init__(self)


class SamplePDF(object):
    """Generate samples from a PDF given by a function (as in the original)."""

    def __init__(self, pdf, a, b, num_points=1024):
        x = np.linspace(a, b, num_points)
        y = pdf(x)
        Y = np.hstack((0, cumtrapz(y, x)))
        Y /= Y[-1]
        self.interp = interp1d(Y, x, kind='linear')

    def __call__(self):
        return self.rvs()

    def rvs(self):
        return float(self.interp(np.random.rand()))


class PartialAligningRotator(_RotateMixin, _core.PartialAligningRotator):
    """Rotation into a Gaussian-weighted random orientation.

    Constructor args:
        exp_sig_deg: The standard deviation of the canting angle, in degrees.
        random_flip: Randomly flip the particle upside down.
    """

    def __init__(self, exp_sig_deg=40, random_flip=False):
        exp_sig = exp_sig_deg * pi / 180
        # table computed with numpy exactly like the original SamplePDF
        self.beta_sample_py = SamplePDF(lambda x: np.sin(x) * np.exp(-0.5 * (x / exp_sig)**2), 0, pi)
        table = _core.SamplePDF(list(self.beta_sample_py.interp.y), list(self.beta_sample_py.interp.x))
        _core.PartialAligningRotator.__init__(self, float(exp_sig_deg), bool(random_flip), table)

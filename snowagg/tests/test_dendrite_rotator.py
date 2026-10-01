"""Reiter dendrite growth (exact) and rotators/SamplePDF."""
import numpy as np
import pytest
from aggregation import dendrite as rd
from aggregation import rotator as rr

from snowagg import _core


@pytest.mark.parametrize("params", [
    (1.0, 0.35, 0.001, 41, 60),
    (1.0, 0.35, 0.001, 100, 400),
    (0.705, 0.5, 0.0001, 81, 250),
    (2.3, 0.4, 0.002, 64, 150),     # alpha > 2: negative centre weight
    (1.0, 0.35, 0.001, 60, 5000),   # stops on the margin criterion
])
def test_reiter_exact(params):
    alpha, beta, gamma, n, it = params
    g_py, it_py, m_py = rd.generate_dendrite(alpha, beta, gamma, grid_size=n, num_iter=it, verbose=True)
    g_cc, it_cc, m_cc = _core.generate_dendrite(alpha, beta, gamma, n, it)
    assert it_py == it_cc
    assert m_py == m_cc
    assert np.array_equal(g_py, g_cc)


def numpy_beta_table(exp_sig_deg):
    """The SamplePDF table exactly as the Python PartialAligningRotator builds it."""
    ref = rr.PartialAligningRotator(exp_sig_deg=exp_sig_deg)
    sp = ref.beta_sample
    # interp1d keeps the (sorted) CDF in .x and the sample values in .y
    return ref, sp, list(sp.interp.y), list(sp.interp.x)


def test_samplepdf_matches_interp1d():
    ref, sp, x, Y = numpy_beta_table(40)
    table = _core.SamplePDF(x, Y)
    u = np.random.default_rng(1).uniform(size=20000)
    u = np.concatenate([u, np.array(Y[:50]), [0.0]])  # also hit the table knots exactly
    assert all(float(sp.interp(v)) == table(v) for v in u)


def test_samplepdf_builtin_table_close():
    # the C++-computed table (libm exp) is equal to the numpy one within rounding
    _, _, x, Y = numpy_beta_table(40)
    rot = _core.PartialAligningRotator(40.0, False)
    assert np.allclose(rot.beta_sample.Y, Y, rtol=1e-13, atol=1e-16)
    assert np.allclose(rot.beta_sample.x, x, rtol=0, atol=0)


@pytest.mark.parametrize("which", ["uniform", "horizontal", "partial", "partial_flip"])
def test_rotators_consume_same_randoms(which):
    X = np.random.default_rng(3).normal(size=(500, 3)) * 1e-3
    if which == "uniform":
        py, cc = rr.UniformRotator(), _core.UniformRotator()
    elif which == "horizontal":
        py, cc = rr.HorizontalRotator(), _core.HorizontalRotator()
    else:
        flip = which == "partial_flip"
        py = rr.PartialAligningRotator(exp_sig_deg=40, random_flip=flip)
        _, _, x, Y = numpy_beta_table(40)
        cc = _core.PartialAligningRotator(40.0, flip, _core.SamplePDF(x, Y))
    r = _core.Rng()
    np.random.seed(17)
    r.set_state(np.random.get_state())
    for _ in range(50):
        Xp = py.rotate(X.T).T
        Xc = cc.rotate(X, r)
        # rotation matrices differ from numpy/BLAS only by rounding
        assert np.allclose(Xp, Xc, rtol=0, atol=1e-15 * np.abs(X).max() * 8)
        assert r.rand() == np.random.rand()  # same number of draws consumed

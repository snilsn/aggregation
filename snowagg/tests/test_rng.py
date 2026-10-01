"""The C++ Rng reproduces numpy's legacy global RandomState exactly."""
import numpy as np
from scipy import stats

from snowagg import _core


def test_rand_stream():
    r = _core.Rng(1234)
    np.random.seed(1234)
    a = np.array([r.rand() for _ in range(20000)])
    assert np.array_equal(a, np.random.rand(20000))


def test_randint():
    r = _core.Rng(99)
    np.random.seed(99)
    for n in [1, 2, 3, 5, 7, 17, 100, 1000, 12345, 2**31 + 5]:
        for _ in range(300):
            assert r.randint(n) == np.random.randint(n)


def test_gauss_with_cache():
    r = _core.Rng(5)
    np.random.seed(5)
    a = [r.gauss() for _ in range(9999)]
    b = [np.random.randn() for _ in range(9999)]
    assert a == b
    # randn(3) as used by deposition
    np.random.seed(6)
    r.seed(6)
    for _ in range(100):
        assert list(np.random.randn(3)) == [r.gauss(), r.gauss(), r.gauss()]


def test_exponential_matches_scipy_expon():
    r = _core.Rng(11)
    np.random.seed(11)
    for scale in [1e-4, 1e-3, 3.3e-3]:
        for _ in range(500):
            assert r.standard_exponential() * scale + 0 == stats.expon(scale=scale).rvs()


def test_state_exchange():
    np.random.seed(7)
    np.random.rand(3)
    np.random.randn()  # leaves a cached gaussian
    r = _core.Rng()
    r.set_state(np.random.get_state())
    assert r.gauss() == np.random.randn()
    assert r.rand() == np.random.rand()
    np.random.set_state(r.get_state())
    assert r.rand() == np.random.rand()


def test_regeneration_boundary():
    # crosses several 624-word regenerations with mixed call types
    r = _core.Rng(2024)
    np.random.seed(2024)
    for i in range(3000):
        k = i % 3
        if k == 0:
            assert r.rand() == np.random.rand()
        elif k == 1:
            assert r.randint(37) == np.random.randint(37)
        else:
            assert r.gauss() == np.random.randn()

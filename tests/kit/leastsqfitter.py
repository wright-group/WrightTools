import numpy as np
import WrightTools as wt


def test_leastsq():
    x = np.linspace(0, 10)
    y = np.linspace(10, 0)
    fit, cov = wt.kit.leastsqfitter([0, 0], x, y, lambda p, x: p[0] * x + p[1])
    assert np.allclose(fit, [-1, 10])
    assert np.allclose(cov, [0, 0])


def test_leastsq_no_corr():
    x = np.linspace(0, 10)
    y = np.linspace(10, 0)
    # The third parameter does not determine output, this caused an exception in wt <= 3.2.1
    fit, cov = wt.kit.leastsqfitter([0, 0, 0], x, y, lambda p, x: p[0] * x + p[1])
    assert np.allclose(fit, [-1, 10, 0])
    assert np.allclose(cov, [0, 0, 0])

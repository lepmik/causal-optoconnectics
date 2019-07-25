import pytest
import numpy as np
from causal_optoconnectics.core import raised_cosine, calculate_regressors


def test_log_raised_cosine():
    time, bases, centers = raised_cosine(11, .01, np.array([0, 1003]), 1)
    # all sum to one until the last peak
    assert all(bases[:1003].sum(1) - 1 < 1e-15)


def test_linear_raised_cosine():
    time, bases, centers = raised_cosine(11, .01, np.array([0, 1003]), 1, stretching='linear')
    # all sum to one until the last peak
    assert all(bases[:1003].sum(1) - 1 < 1e-15)


def test_calculate_regressors():
    s = np.array([1, 2, 3, 4, 5,  6]).astype(float)

    x = np.array([0.1, 0.2, 1.2, 1.3, 2.1, 2.4, 4.4, 4.5, 5, 5.05, 6.01])

    y = np.array([0.1, 0.2, 1.2, 1.22, 1.23, 1.3, 2.1, 2.2, 2.4, 3, 3.3, 5, 5.1, 6.3])

    Z_true = np.array([-0.8, -0.7, -0.6, -1.6 ,0.  ,-0.95])
    X_true = np.array([0.2 , 0.1 , 1.4 , 0.4 , 0.05, 0.01])
    Y_true = np.array([3, 2, 1, 0, 2, 1])

    Z, X, Y = calculate_regressors(x, y, s, 0.15, 0.15)

    assert np.allclose(Z, Z_true), (Z, Z_true)
    assert np.allclose(X, X_true), (X, X_true)
    assert np.array_equal(Y, Y_true), (Y, Y_true)

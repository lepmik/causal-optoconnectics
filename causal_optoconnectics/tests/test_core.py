import pytest
import numpy as np
from causal_optoconnectics.core import find_first_response_spike, find_response_spikes


def test_find_first_response_spike():
    s = np.array([1, 2, 3, 4, 5,  6]).astype(float)

    x = np.array([0.1, 0.2, 1.2, 1.3, 2.1, 2.4, 4.4, 4.5, 5, 5.05, 6.01])

    y = np.array([0.1, 0.2, 1.2, 1.22, 1.23, 1.3, 2.1, 2.2, 2.4, 3, 3.3, 5, 5.1, 6.3])

    Z_true = np.array([-0.8, -0.7, -0.6, -1.6 ,0.  ,-0.95])
    X_true = np.array([0.2 , 0.1 , 1.4 , 0.4 , 0.05, 0.01])
    Y_true = np.array([0.2, 0.1, 0.3, 1. , 0.1, 0.3])

    Z, X, Y, _ = find_first_response_spike(x, y, s)

    assert np.allclose(Z, Z_true), (Z, Z_true)
    assert np.allclose(X, X_true), (X, X_true)
    assert np.allclose(Y, Y_true), (Y, Y_true)


def test_find_spikes():
    s = np.array([1, 2, 3, 4, 5])
    a = np.array([
        .1, .99, # before 1 (Z False)
        1.11, 1.12, # after 1
        2, # before 2 (Z True)
        2.13, # after 2
        3.14, 3.15, # after 3
        3.999, # before 4 (Z True)
        4.16, # after 4
        5.17# before 1
    ])
    c = np.array([
        .2,
        1.13, 1.23,
        2.34,
        3.2,
        4.1,
        5.5])
    Z, X, Y = find_response_spikes(a, c, s, dt=.3, dz=0.01)
    X_true = [
        np.array([0.11, 0.12]),
        np.array([0.13]),
        np.array([0.14, 0.15]),
        np.array([0.16]),
        np.array([0.17])
    ]
    Y_true = [
        np.array([0.13, 0.23]),
        np.array([]),
        np.array([0.2]),
        np.array([0.1]),
        np.array([])
    ]
    assert np.array_equal(Z, np.array([False,  True, False,  True, False]))
    assert all([np.allclose(X[i], X_true[i]) for i in range(len(X))])
    assert all([np.allclose(Y[i], Y_true[i]) for i in range(len(Y))])

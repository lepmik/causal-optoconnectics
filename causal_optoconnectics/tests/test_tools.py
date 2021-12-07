import pytest
import numpy as np
from causal_optoconnectics.tools import compute_trials


def test_compute_trials_from_spike_times():
    s = np.array([1, 2, 3, 4, 5])
    a = np.array([
        .1, .99, # before 1 (Z True)
        1.01, 1.02, # after 1 (X True)
        2, # before 2 (Z True)
        2.13, # after 2 (X False)
        3.14, 3.15, # after 3
        3.999, # before 4 (Z True)
        4.01, # after 4
        5.17# before 1
    ])
    c = np.array([
        .2,
        1.03, 1.04, # after 1 (Y True)
        2.34,
        3.2,
        4.1,
        5.03])
    # convert to indexes
    s, a, c = (s*100).astype(int), (a*100).astype(int), (c*100).astype(int)
    a = a.repeat(2).reshape((len(a), 2))
    a[:,0] = 0 # event id
    c = c.repeat(2).reshape((len(c), 2))
    c[:,0] = 1 # event id
    s = s.repeat(2).reshape((len(s), 2))
    s[:,0] = 2 # event id
    events = np.concatenate([s, a, c], 0)
    sort_idxs = np.argsort(events[:,1], 0)
    events = events[sort_idxs, :]
    trials = compute_trials(events, neurons=2, stim_index=2, n1=-10, n2=10)
    Z = trials[0][:, 9:11].sum(1).astype(bool)
    X = trials[0][:, 11:12].sum(1).astype(bool)
    Y = trials[1][:, 13:14].sum(1).astype(bool)

    assert np.array_equal(Z, np.array([True,  True, False,  True, False]))
    assert np.array_equal(X, np.array([True,  False, False,  True, False]))
    assert np.array_equal(Y, np.array([True,  False, False,  False, True]))

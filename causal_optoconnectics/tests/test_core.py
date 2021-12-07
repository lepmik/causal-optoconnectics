import pytest
import numpy as np
from causal_optoconnectics.core import Connectivity
from causal_optoconnectics.tools import compute_trials


def test_connectivity():
    stimulus = np.arange(10, 510, 10).repeat(2).reshape((50, 2)).astype(float)
    stimulus[:,0] = 3 # event id
    A = stimulus + 1
    A[:,0] = 0 # event id
    B = stimulus + 1
    B[:,0] = 1 # event id
    C = stimulus + 3
    C[:,0] = 2 # event id
    # Neuron A only responds 50% of the time because it spikes before
    # stimulus
    A[1::2, 1] -= 1
    # Neuron B only responds 80% of the time, but C follows
    B[::5, 1] -= 1
    C[::5, 1] -= 1
    # Confounding factor shuts of A, B and C together, this makes the
    # OLS estimator fail
    A[::6] = np.nan
    B[::6] = np.nan
    C[::6] = np.nan
    # Estimate hit rate
    idxs = np.arange(50)
    hit_rate_A = 1 - len(np.unique(np.concatenate((idxs[1::2], idxs[::6])))) / 50
    hit_rate_B = 1 - len(np.unique(np.concatenate((idxs[::6], idxs[::5])))) / 50
    # Now we combine the activity together in a event array
    events = np.concatenate([stimulus, A, B, C], 0)
    # Remove nans
    events = events[np.isfinite(events[:,1])]
    sort_idxs = np.argsort(events[:,1], 0)
    events = events[sort_idxs, :].astype(int)
    trials = compute_trials(events, neurons=3, stim_index=3, n1=-2, n2=4)
    assert trials[0].shape == (50, 6)
    # Compute connectivity
    # parameters are indices relative to the range n2 - n1,
    # i.e. 6 is the last index in the trial bins
    params = dict(x1=3, x2=4, y1=5, y2=6, z1=1, z2=3)
    AC = Connectivity(pre=trials[0], post=trials[2], params=params)
    BC = Connectivity(pre=trials[1], post=trials[2], params=params)
        # First we look at the raw values
    AC.compute(rectify=False)
    BC.compute(rectify=False)
    assert round(AC.beta_iv, 3) == -0.438
    assert round(AC.beta_brew, 3) == 0.012
    assert round(AC.beta_ols, 3) == 0.224
    assert round(AC.beta_iv_did, 3) == -0.122
    assert round(AC.beta_brew_did, 3) == 0.025
    assert round(AC.beta_ols_did, 3) == 0.184
    assert round(AC.hit_rate, 2) == round(hit_rate_A, 2)

    assert round(BC.beta_iv, 2) == 1.00
    assert round(BC.beta_brew, 2) == 1.00
    assert round(BC.beta_ols, 2) == 1.00
    assert round(BC.beta_iv_did, 2) == 1.00
    assert round(BC.beta_brew_did, 2) == 2.00
    assert round(BC.beta_ols_did, 2) == 1.47
    assert round(BC.hit_rate, 2) == round(hit_rate_B, 2)

    AC.compute(rectify=True)
    BC.compute(rectify=True)
    assert round(AC.beta_iv, 3) == 0.000
    assert round(AC.beta_brew, 3) == 0.012
    assert round(AC.beta_ols, 3) == 0.224
    assert round(AC.beta_iv_did, 3) == 0.000
    assert round(AC.beta_brew_did, 3) == 0.025
    assert round(AC.beta_ols_did, 3) == 0.184
    assert round(AC.hit_rate, 2) == round(hit_rate_A, 2)

    assert round(BC.beta_iv, 2) == 1.00
    assert round(BC.beta_brew, 2) == 1.00
    assert round(BC.beta_ols, 2) == 1.00
    assert round(BC.beta_iv_did, 2) == 1.00
    assert round(BC.beta_brew_did, 2) == 2.00
    assert round(BC.beta_ols_did, 2) == 1.47
    assert round(BC.hit_rate, 2) == round(hit_rate_B, 2)

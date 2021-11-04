import numpy as np
import scipy.stats as st


def compute_stim_response(stim_times, spikes, x1, x2):
    '''Calculate upstream spike time before and after stimulus and downstream
    count. Responses in are counted within
    `y >= stim_times + y_mu - y_sigma`
    and `y < stim_times + y_mu + y_sigma`.

    Parameters
    ----------
    spikes : array
        Spike times
    stim_times : array
        Stimulation onset times
    x1 : array
        window lower bound
    x2 : array
        window upper bound
    Returns
    -------
    Y : array
        Times (x1 <= s < x2) relative to stimulus times
    '''
    stim_win = np.insert(
        stim_times + x1,
        np.arange(len(stim_times)) + 1,
        stim_times + x2)
    src_y = np.searchsorted(spikes, stim_win, side='right')
    cnt_y = np.diff(src_y.reshape((int(len(src_y) / 2), 2)))
    Y = cnt_y.flatten()
    return Y.astype(bool).astype(int)



def find_response_spikes(x, y, s, z1, z2, dt):
    '''Calculate upstream spike time before and after stimulus and downstream
    count. Responses in are counted within
    `y >= stim_times + y_mu - y_sigma`
    and `y < stim_times + y_mu + y_sigma`.

    Parameters
    ----------
    x : array
        Upstream spike times
    y : array
        Downstream spike times
    s : array
        Stimulation onset times
    Returns
    -------
    Z : array
        Upstream times (< 0) before stimulus (referenced at 0)
    X : array
        Upstream times (> 0) after stimulus (referenced at 0)
    Y : array
        Downstream times (> 0) after stimulus (referenced at 0)
    '''
    s = s.astype(float)
    X, Y, Z = [], [], []
    for t in s:
        # searchsorted:
        # left	a[i-1] < v <= a[i]
        # right	a[i-1] <= v < a[i]
        # t - dz < z <= t
        idx_z = np.searchsorted(x, [t - z1, t - z2], side='right')
        Z.append(np.diff(idx_z) > 0)
        # t < x <= t + dt
        idx_x = np.searchsorted(x, [t, t + dt], side='right')
        X.append(x[idx_x[0]: idx_x[1]] - t)
        # t < y <= t + dt
        idx_y = np.searchsorted(y, [t, t + dt], side='right')
        Y.append(y[idx_y[0]: idx_y[1]] - t)
    Z = np.array(Z, dtype=bool).ravel()
    X = np.array(X)
    Y = np.array(Y)
    return Z, X, Y


class Connectivity:
    def __init__(self, pre, post, x1, x2, y1, y2, z1, z2):
        '''
        pre/post are trials of size n_trials x n_bins
        assumes that the stimulation starts on the middle bin
        x1, x2, y1, y2, z1, z2 are bin indices for where
        '''
        n_bins = pre.shape[1]
        assert n_bins % 2 == 0
        #stim_idx = int(n_bins / 2)
        n_response = y2-y1

        x = pre[:, x1:x2].sum(1).astype(bool)
        y = post[:, y1:y2].sum(1).astype(bool)
        z = pre[:, z1:z2].sum(1).astype(bool)

        #y0 = post[:, stim_idx-n_response:stim_idx].sum(1).astype(bool)
        y0 = post[:,y1-n_response:y2-n_response].sum(1).astype(bool)

        self.x, self.y, self.z, self.y0 = x, y, z, y0

        y_refractory = (y*z).sum() / z.sum()

        y_response = (y*x).sum() / x.sum()

        y_nospike = (y*(1-x)).sum() / (1-x).sum()

        y0_refractory = (y0*z).sum() / z.sum()

        y0_response = (y0*x).sum() / x.sum()

        y0_nospike = (y0*(1-x)).sum() / (1-x).sum()

        # standard iv
        self.beta_iv = y_response - y_refractory
        # OLS
        self.beta = y_response - y_nospike

        # DiD iv
        self.beta_iv_did = self.beta_iv - (y0_response - y0_refractory)
        # OLS
        self.beta_did = self.beta - (y0_response - y0_nospike)

        self.hit_rate = x.mean()

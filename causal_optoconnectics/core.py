import numpy as np
import scipy.stats as st
import scipy.integrate as si
from sklearn.linear_model import LogisticRegression # must come before nest import


def find_first_response_spike(x, y, s):
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

    src_x = np.searchsorted(x, s, side='right')
    src_y = np.searchsorted(y, s, side='right')

    remove_idxs, = np.where((src_x==len(x)) | (src_y==len(y)))
    src_x = np.delete(src_x, remove_idxs)
    src_y = np.delete(src_y, remove_idxs)
    s = np.delete(s, remove_idxs)

    X = x[src_x] - s
    Y = y[src_y] - s
    Z = x[src_x-1] - s

    return Z, X, Y, s


def find_response_spikes(x, y, s, dt, dz):
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
        idx_z = np.searchsorted(x, [t - dz, t], side='right')
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


def causal_connectivity(x, y, s, x1, x2, y1, y2, dt, dz):
    Z, X, Y = find_response_spikes(x, y, s, dt, dz)
    #X = [t for trial in X for t in trial]
    X = np.array([any((t >= x1) & (t <= x2)) for t in X])
    Y_0 = [t for trial in Y[X==1] for t in trial]
    Y_1 = [t for trial in Y[Z==1] for t in trial]
    #px = st.gaussian_kde(X, .01)
    py0 = st.gaussian_kde(Y_0, .01)
    py1 = st.gaussian_kde(Y_1, .01)

    #Px = px.integrate_box_1d(x1, x2)
    Py0 = py0.integrate_box_1d(y1, y2)
    Py1 = py1.integrate_box_1d(y1, y2)
    d = y2 - y1
    Py00 = py0.integrate_box_1d(dt - d, dt)
    Py11 = py1.integrate_box_1d(dt - d, dt)
    return Py0 - Py00 - Py1 + Py11


def probable_connectivity(x, y, s, x1, x2, y1, y2, dt):
    _, X, Y = find_response_spikes(x, y, s, dt, 1)
    #X = [t for trial in X for t in trial]
    X = np.array([any((t >= x1) & (t <= x2)) for t in X])
    Y_0 = [t for trial in Y[X==1] for t in trial]
    Y_1 = [t for trial in Y[X==0] for t in trial]
    #px = st.gaussian_kde(X, .01)
    py0 = st.gaussian_kde(Y_0, .01)
    py1 = st.gaussian_kde(Y_1, .01)

    #Px = px.integrate_box_1d(x1, x2)
    Py0 = py0.integrate_box_1d(y1, y2)
    Py1 = py1.integrate_box_1d(y1, y2)
    return Py0 - Py1


def causal_connectivity_mean(x, y, s, x1, x2, y1, y2, dt, dz):
    Z, X, Y = find_response_spikes(x, y, s, dt, dz)
    #X = [t for trial in X for t in trial]
    X = np.array([any((t >= x1) & (t <= x2)) for t in X])
    Y_1 = np.array([any((t >= y1) & (t <= y2)) for t in Y])
    d = y2 - y1
    Y_0 = np.array([any((t >= dt - d) & (t <= dt)) for t in Y])
    assert X[Z==1].mean() < 1e-10
    return Y_1[X==1].mean() - Y_1[Z==1].mean() - (Y_0[X==1].mean() - Y_0[Z==1].mean())


def probable_connectivity_mean(x, y, s, x1, x2, y1, y2, dt):
   _, X, Y = find_response_spikes(x, y, s, dt, 1)
   X = np.array([any((t >= x1) & (t <= x2)) for t in X])
   Y = np.array([any((t >= y1) & (t <= y2)) for t in Y])

   return Y[X==1].mean() - Y[X==0].mean()

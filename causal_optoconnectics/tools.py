import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize_scalar
from .core import Connectivity


def poisson_continuity_correction(n, observed):
    """Correct for continuity.

    Parameters
    ----------
    n : array
        Likelihood to observe n or more events
    observed : array
        Rate of Poisson process

    Returns
    -------
    array
        Corrected array.

    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.

    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    """
    if n.ndim == 0:
        n = np.array([n])
    assert n.ndim == 1
    from scipy.stats import poisson
    assert np.all(n >= 0)
    result = np.zeros(n.shape)
    if n.shape != observed.shape:
        observed = np.repeat(observed, n.size)
    for i, (n_i, rate) in enumerate(zip(n, observed)):
        if n_i == 0:
            result[i] = 1.
        else:
            rates = [poisson.pmf(j, rate) for j in range(n_i)]
            result[i] = 1 - np.sum(rates) - 0.5 * poisson.pmf(n_i, rate)
    return result


def hollow_kernel(kernlen, width, hollow_fraction=0.6):
    '''Returns a hollow kernel normalized to it's sum

    Parameters
    ----------
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.

    Returns
    -------
    kernel : array

    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    '''
    from scipy.signal import gaussian
    assert kernlen % 2 == 1
    kernel = gaussian(kernlen, width)
    kernel[int(kernlen / 2.)] *= (1 - hollow_fraction)
    return kernel / sum(kernel)


def histogram(val, bins, density=False):
    """Fast histogram


    Parameters
    ----------
    val : array
        Values to be counted in bins.
    bins : array
        Bins to count in.
    density : bool
        Normalize to probability density.

    Returns
    -------
    counts, bins : array, array

    Note
    ----
    Assumes `val`, `bins` are sorted,
    bins increase monotonically and uniformly,
    `all(bins[0] <= v <= bins[-1] for v in val)`

    """
    result = np.zeros(len(bins) - 1).astype(int)
    search = np.searchsorted(bins, val, side='right')
    cnt = np.bincount(search)[1:len(result)]
    result[:len(cnt)] = cnt
    if density:
        db = np.array(np.diff(bins), float)
        return result / db / result.sum(), bins
    return result, bins


def decompress_events(events, n_neurons, n_time_steps):
    """Decompress events from indices to boolean array.

    Parameters
    ----------
    events : array, (n_events, 2)
        Array of unit ids in first column and corresponding spike-time indices
        on second column.
    n_neurons : int
        Number of neurons.
    n_time_steps : int
        Number of time steps.

    Returns
    -------
    array, (n_neurons, n_time_steps)
        Boolean indicator of events

    """
    x = np.zeros((n_neurons, n_time_steps))
    x[events[:,0], events[:,1]] = 1
    return x


def compute_trials(events, neurons, stim_index, n1=-10, n2=10):
    """Compute trials from spike indices.

    Parameters
    ----------
    events : array, (n_events, 2)
        Array of unit ids in first column and corresponding spike-time indices
        on second column. Assumes neurons are indexed 0:n_neurons.
    neurons : int or array
        Neurons to compute,
        if int assumes neurons in `events` are indexed 0:n_neurons.
    stim_index : type
        The index in `events` that indicate stimulus onsets.
    n1 : type
        Number of steps relative to stim onset (the default is -10).
    n2 : type
        Number of steps relative to stim onset (the default is 10).

    Returns
    -------
    dict
        Trials from each neuron.

    Examples
    --------
    >>> import numpy as np
    >>> events = np.random.randint(0, 10, (100,2))
    >>> trials = compute_trials(events, 9, 10)

    """
    neurons = neurons if not isinstance(neurons, int) else range(neurons)
    from collections import defaultdict
    stim_indices = events[events[:, 0]==stim_index, 1]
    trials = defaultdict(list)
    for ni in neurons:
        xx = events[events[:, 0]==ni, 1]
        xxx = np.zeros((len(stim_indices), abs(n1 - n2)))
        for ii, i in enumerate(stim_indices):
            idx = np.searchsorted(xx, [i + n1, i + n2], side='left')
            xxx[ii, xx[idx[0]:idx[1]]-i-n1] = 1
        trials[ni] = xxx
    return trials


def compute_trials_multi(events_list, neurons, stim_index, n1=-10, n2=10):
    """Compute trials from spike indices from multiple datasets, using
    multiprocessing.

    Parameters
    ----------
    events_list : list, [(n_events, 2), ...]
        List with multiple arrays of unit ids in first column and corresponding spike-time indices
        on second column. Assumes neurons are indexed 0:n_neurons.
    neurons : int or array
        Neurons to compute,
        if int assumes neurons in `events` are indexed 0:n_neurons.
    stim_index : type
        The index in `events` that indicate stimulus onsets.
    n1 : type
        Number of steps relative to stim onset (the default is -10).
    n2 : type
        Number of steps relative to stim onset (the default is 10).

    Returns
    -------
    dict
        Trials from each neuron.

    Examples
    --------
    >>> import numpy as np
    >>> events = [np.random.randint(0, 10, (100,2)) for _ in range(10)]
    >>> trials = compute_trials_multi(events, 9, 10)

    """
    neurons = neurons if not isinstance(neurons, int) else range(neurons)
    import multiprocessing
    from collections import defaultdict
    from functools import partial
    with multiprocessing.Pool() as p:
        samples = p.map(partial(
            compute_trials, neurons=neurons, stim_index=stim_index,
            n1=n1, n2=n2), events_list)

    trials = defaultdict(list)
    for ni in neurons:
        for sample in samples:
            trials[ni].append(sample[ni])
        trials[ni] = np.concatenate(trials[ni])

    return trials


def joint_probability(x, y):
    """Compute the joint probability of (binary) x and y using multiprocessing.
    Same as `np.matmul(y.T, x) / len(x)`

    Parameters
    ----------
    x : array (2D)
        Parameter `x`.
    y : array (2D)
        Parameter `y`.

    Returns
    -------
    array
        Joint probability.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randint(0, 2, (10,10))
    >>> y = np.random.randint(0, 2, (10,10))
    >>> jp = joint_probability(x, y)
    >>> assert (np.matmul(y.T, x) / len(x) == jp).all()
    """
    import multiprocessing
    from joblib import Parallel, delayed

    num_cores = multiprocessing.cpu_count()

    res = Parallel(n_jobs=num_cores)(delayed(lambda i: (x.T * y[:,i]).mean(1))(i) for i in range(x.shape[1]))
    return np.array(res)


def conditional_probability(x, y):
    """Compute the conditional probability of (binary) x and y using
    multiprocessing.

    Parameters
    ----------
    x : array (2D)
        Parameter `x`.
    y : array (2D)
        Parameter `y`.

    Returns
    -------
    array
        conditional probability.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randint(0, 2, (10,10))
    >>> y = np.random.randint(0, 2, (10,10))
    >>> cp = conditional_probability(x, y)
    """
    joint = joint_probability(x, y)
    cond = joint / joint.sum(0)
    cond[np.isnan(cond)] = 0
    return cond


def roll_pad(x, i, axis=1):
    """Roll array padding with zeros.

    Parameters
    ----------
    x : array, (1D or 2D)
        Array to roll.
    i : int
        Number of steps to roll. Negative index is interpreted as right roll.
    axis : int
        Axis to roll (the default is 1).

    Returns
    -------
    array
        Rolled array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0,1,1])
    >>> rx = roll_pad(x, 1)
    >>> assert (rx == np.array([1,1,0])).all()

    """
    if axis != 1:
        raise NotImplementedError
    if x.ndim == 2:
        if i == 0:
            return x
        elif i > 0:
            return np.pad(x, ((0, 0),(i, 0)))[:, :-i]
        else:
            return np.pad(x, ((0, 0), (0, abs(i))))[:, abs(i):]
    elif x.ndim == 1:
        if i == 0:
            return x
        elif i > 0:
            return np.pad(x, (i, 0))[:-i]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def process_metadata(sources, targets, W, stim_index, ignore_self_connection=True):
    pairs = []
    for i in sources:
        for j in targets:
            if i==j and ignore_self_connection:
                continue
            pairs.append(process(i, j, W, stim_index, meta_only=True))
    return pairs


def process(source, target, W, stim_index, params=None, trials=None, n_trials=None, meta_only=False, compute_values=True):
    if trials is not None:
        pre, post = trials[source], trials[target]
        n_trials = len(pre) if n_trials is None else n_trials
        pre, post = pre[:n_trials], post[:n_trials]
    else:
        pre, post = None, None

    result = {
        'source': source,
        'target': target,
        'pair': (source, target),
        'weight': W[source, target, 0],
        'source_stim': W[stim_index, source, 0] > 0,
        'source_stim_strength': W[stim_index, source, 0],
        'target_stim': W[stim_index, target, 0] > 0,
    }
    if meta_only:
        return result

    conn = Connectivity(
        pre,
        post,
        params
    )
    if compute_values:
        conn.compute()
    result.update(conn.__dict__)
    result.update(params)
    return result


def compute_time_dependence(i, j, step=10000):
    pre, post = trials[i], trials[j]
    results = []
    start = 0
    for stop in tqdm(range(step, len(pre) + step, step)):
        results.append(process(i, j, stop))
    return results


def error(a, df, key):
    return df['weight'] - a * df[key]


def error_norm(a, df, key):
    return norm(error(a, df, key))


def min_error(df, key):
    return minimize_scalar(error_norm, args=(df, key))


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

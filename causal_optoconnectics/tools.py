"""
This module contain all tools

"""
import numpy as np
from scipy.linalg import norm
from scipy.optimize import minimize_scalar
from .core import Connectivity


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
    >>> trials = compute_trials(events=events, neurons=9, stim_index=9)

    """
    neurons = neurons if not isinstance(neurons, int) else range(neurons)
    from collections import defaultdict
    stim_indices = events[events[:, 0]==stim_index, 1]
    trials = {}
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
    >>> assert (rx == np.array([0,0,1])).all(), rx

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


def process(source, target, W, stim_index, params=None, trials=None, n_trials=None, meta_only=False, compute_values=True, rectify=False):
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
        conn.compute(rectify=rectify)
    result.update(conn.__dict__)
    result.update(params)
    return result


def reduce_sum(dfs):
    keys = [
        "yz_sum",
        "z_sum",
        "yzinv_sum",
        "zinv_sum",
        "yx_sum",
        "x_sum",
        "yxinv_sum",
        "xinv_sum",
        "xz_sum",
        "xzinv_sum",
        "y0z_sum",
        "y0zinv_sum",
        "y0x_sum",
        "y0xinv_sum",
        "x0z_sum",
        "x0zinv_sum",
        'n_trials'
    ]
    result = dfs[0].copy()
    for key in keys:
        if key != 'n_trials':
            result[key] = sum([df[key].values for df in dfs])
        else:
            assert (dfs[0][key].values == dfs[0][key].values).all()
            result[key] = sum([df[key].values[0] for df in dfs])
    return result


def compute_connectivity_from_sum(row):
    conn = Connectivity()
    conn.__dict__.update(row.to_dict())
    conn.compute()
    result = conn.__dict__
    return result


def compute_time_dependence(i, j, step=10000):
    pre, post = trials[i], trials[j]
    results = []
    start = 0
    for stop in tqdm(range(step, len(pre) + step, step)):
        results.append(process(i, j, stop))
    return results


def error(a, df, key):
    return df[key] * a - df['weight']


def error_norm(a, df, key):
    return norm(error(a, df, key))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def min_error(df, key):
    return np.sin(angle(df[key].values, df['weight'].values))


# def min_error(df, key):
#     min_l2 = minimize_scalar(error_norm, args=(df, key), bounds=[0,np.inf]).fun
#     return min_l2 / norm(df['weight'])


def rsquared(df, key):
    import statsmodels.api as sm
    _x = df[key]
    _y = df['weight']
    X = np.c_[np.ones(len(_x)), _x]
    results = sm.OLS(_y, X).fit()
    return results.rsquared

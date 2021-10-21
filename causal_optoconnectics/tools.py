import numpy as np


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


def compute_trials_multi(events, neurons, stim_index, n1=-10, n2=10):
    """Compute trials from spike indices from multiple datasets, using
    multiprocessing.

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
    >>> events = [np.random.randint(0, 10, (100,2)) for _ in range(10)]
    >>> trials = compute_trials_multi(events, 9, 10)

    """
    neurons = neurons if not isinstance(neurons, int) else range(neurons)
    import multiprocessing
    from collections import defaultdict
    from functools import partial
    with multiprocessing.Pool() as p:
        samples = p.map(partial(
            compute_trials, n_neurons=n_neurons, stim_index=stim_index,
            n1=n1, n2=n2), events)

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

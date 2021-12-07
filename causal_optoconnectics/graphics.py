"""
This module contain all graphics and plotting

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def probplot(prob, sigma=1, xmin=-10, xmax=10, ymin=-10, ymax=10, ax=None, colorbar=True, grid=False):
    """Plot a conditional probability matrix.

    Parameters
    ----------
    prob : ndarray
        Description of parameter `prob`.
    sigma : float
        Smoothing parameter (the default is 1).
    xmin : int
        Start of xrange (the default is -10).
    xmax : int
        Stop of xrange `xmax` (the default is 10).
    ymin : int
        start of yrange `ymin` (the default is -10).
    ymax : int
        Stop of yrange `ymax` (the default is 10).
    ax : matplotlib.axes.Axes
        Axes to plot on `ax` (the default is None).
    colorbar : bool
        Make colorbar or not `colorbar` (the default is True).
    grid : bool, str
        Wheter or not to make a grid, if not False should be string
        'full' or 'zero' (the default is False).

    Returns
    -------
    matplotlib.axes.Axes
        Description of returned object.

    Examples
    --------
    >>> from causal_optoconnectics.tools import compute_trials, conditional_probability
    >>> stimulus = np.arange(10, 510, 10).repeat(2).reshape((50, 2)).astype(float)
    >>> stimulus[:,0] = 3 # event id
    >>> A = stimulus + 1
    >>> A[:,0] = 0 # event id
    >>> B = stimulus + 1
    >>> B[:,0] = 1 # event id
    >>> C = stimulus + 3
    >>> C[:,0] = 2 # event id
    >>> # Neuron A only responds 50% of the time because it spikes before
    >>> # stimulus
    >>> A[1::2, 1] -= 1
    >>> # Neuron B only responds 80% of the time, but C follows
    >>> B[::5, 1] -= 1
    >>> C[::5, 1] -= 1
    >>> # Confounding factor shuts of A, B and C together, this makes the
    >>> # OLS estimator fail
    >>> A[::6] = np.nan
    >>> B[::6] = np.nan
    >>> C[::6] = np.nan
    >>> # Now we combine the activity together in a event array
    >>> events = np.concatenate([stimulus, A, B, C], 0)
    >>> # Remove nans
    >>> events = events[np.isfinite(events[:,1])]
    >>> sort_idxs = np.argsort(events[:,1], 0)
    >>> events = events[sort_idxs, :].astype(int)
    >>> trials = compute_trials(events, neurons=3, stim_index=3, n1=-2, n2=4)
    >>> prob_AC = conditional_probability(trials[0], trials[2])
    >>> ax = probplot(prob_AC, 0, -2, 4, -2, 4)

    """
    from scipy.ndimage.filters import gaussian_filter
    if ax is None:
        fig, (ax, cax) = plt.subplots(
            1, 2,
            gridspec_kw={'width_ratios':[1,0.05], 'wspace': 0.01},
            figsize=(6,5), dpi=150)

    # X, Y = np.meshgrid(np.arange(xmin,xmax), np.arange(ymin,ymax))
    if not sigma: # just to be sure the filter does nothing
        im = ax.imshow(prob, origin='lower')
    else:
        im = ax.imshow(np.rot90(gaussian_filter(prob, sigma)))
    ax.set_xticks(np.arange(prob.shape[1]))
    ax.set_xticklabels(np.arange(xmin, xmax), rotation=45)
    ax.set_yticks(np.arange(prob.shape[0]))
    ax.set_yticklabels(np.arange(ymin,ymax))
    if colorbar:
        plt.colorbar(im, cax=cax)
    ax.set_aspect((ymax-ymin) / (xmax-xmin))
    if grid == 'full':
        ax.grid(True, lw=0.5, alpha=0.5)
    elif grid == 'zero':
        ax.axvline(prob.shape[1] / 2, color='grey', alpha=0.5, lw=0.5)
        ax.axhline(prob.shape[0] / 2, color='grey', alpha=0.5, lw=0.5)
    sns.despine()
    return ax


def regplot(x, y, data, model=None, ci=95., scatter_color=None, model_color='k', ax=None,
            scatter_kws={}, regplot_kws={}, cmap=None, cax=None, clabel=None,
            xlabel=True, ylabel=True, colorbar=False, **kwargs):
    """Plot data and a linear regression model fit.

    This function is adapted from seaborn.regplot, basically, only to make
    the regression model available.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.
    data : type
        Description of parameter `data`.
    model : type
        Description of parameter `model` (the default is None).
    ci : type
        Description of parameter `ci` (the default is 95.).
    scatter_color : type
        Description of parameter `scatter_color` (the default is None).
    model_color : type
        Description of parameter `model_color` (the default is 'k').
    ax : type
        Description of parameter `ax` (the default is None).
    scatter_kws : type
        Description of parameter `scatter_kws` (the default is {}).
    regplot_kws : type
        Description of parameter `regplot_kws` (the default is {}).
    cmap : type
        Description of parameter `cmap` (the default is None).
    cax : type
        Description of parameter `cax` (the default is None).
    clabel : type
        Description of parameter `clabel` (the default is None).
    xlabel : type
        Description of parameter `xlabel` (the default is True).
    ylabel : type
        Description of parameter `ylabel` (the default is True).
    colorbar : type
        Description of parameter `colorbar` (the default is False).
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    Examples
    --------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    if model is None:
        import statsmodels.api as sm
        model = sm.OLS
    from seaborn import utils
    from seaborn import algorithms as algo
    if ax is None:
        fig, ax = plt.subplots()
    _x = data[x]
    _y = data[y]
    grid = np.linspace(_x.min(), _x.max(), 100)

    X = np.c_[np.ones(len(_x)), _x]
    G = np.c_[np.ones(len(grid)), grid]

    results = model(_y, X, **kwargs).fit()

    def reg_func(xx, yy):
        yhat = model(yy, xx, **kwargs).fit().predict(G)
        return yhat
    yhat = results.predict(G)
    yhat_boots = algo.bootstrap(
        X, _y, func=reg_func, n_boot=1000, units=None)
    err_bands = utils.ci(yhat_boots, ci, axis=0)
    ax.plot(grid, yhat, color=model_color, **regplot_kws)
    sc = ax.scatter(_x, _y, c=scatter_color, **scatter_kws)
    ax.fill_between(grid, *err_bands, facecolor=model_color, alpha=.15)
    if colorbar:
        cb = plt.colorbar(mappable=sc, cax=cax, ax=ax)
        cb.ax.yaxis.set_ticks_position('right')
        if clabel: cb.set_label(clabel)

    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
    return results


def scatterplot(x, y, data, scatter_color=None, ax=None,
            cmap=None, cax=None, clabel=None,
            xlabel=True, ylabel=True, colorbar=False, **kwargs):
    """Scatter plot data.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y : type
        Description of parameter `y`.
    data : type
        Description of parameter `data`.
    scatter_color : type
        Description of parameter `scatter_color` (the default is None).
    ax : type
        Description of parameter `ax` (the default is None).
    cmap : type
        Description of parameter `cmap` (the default is None).
    cax : type
        Description of parameter `cax` (the default is None).
    clabel : type
        Description of parameter `clabel` (the default is None).
    xlabel : type
        Description of parameter `xlabel` (the default is True).
    ylabel : type
        Description of parameter `ylabel` (the default is True).
    colorbar : type
        Description of parameter `colorbar` (the default is False).
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    Examples
    --------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """

    if ax is None:
        fig, ax = plt.subplots()
    _x = data[x]
    _y = data[y]

    sc = ax.scatter(_x, _y, c=scatter_color, **kwargs)
    if colorbar:
        cb = plt.colorbar(mappable=sc, cax=cax, ax=ax)
        cb.ax.yaxis.set_ticks_position('right')
        if clabel: cb.set_label(clabel)

    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)

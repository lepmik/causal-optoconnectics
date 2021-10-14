import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def probplot(prob, sigma=1, xmin=-10, xmax=10, ymin=-10, ymax=10, ax=None, colorbar=True, grid=False):
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
        im = ax.imshow(np.rot90(gaussian_filter(prob, sigma)), extent=(xmin, xmax, ymin, ymax))
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

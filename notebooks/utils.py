import numpy as np
from numpy.random import default_rng
import scipy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook as tqdm
import pandas as pd
import pathlib
from collections import defaultdict
import seaborn as sns
from functools import partial, reduce
import ruamel.yaml
yaml = ruamel.yaml.YAML()
from causal_optoconnectics.graphics import regplot, scatterplot, probplot
from causal_optoconnectics.tools import (
    min_error,
    error_norm,
    reduce_sum,
    compute_connectivity_from_sum,
    rsquared
)
from causal_optoconnectics.core import Connectivity

colors = {
    'iv,did': '#e41a1c',
    'iv': '#f781bf',
    'brew,did': '#984ea3',
    'br,did': '#984ea3',
    'br': '#984ea3',
    'ols,did': '#377eb8',
    'ols': '#4daf4a',
    'cch': '#e6ab02'
}

labels = {'did': 'DiD', 'ols': 'OLS', 'iv':'IV', 'brew': 'BR', 'cch': 'CCH', 'br': 'BR'}

default_keys = [
    'beta_ols', 'beta_iv', 'beta_brew', 
    'beta_ols_did', 'beta_iv_did', 'beta_brew_did', 'naive_cch']


def savefig(stem):
    fname = pathlib.Path(f'../paper/graphics/{stem}').with_suffix('.svg')
    plt.savefig(fname, bbox_inches='tight', transparent=True)
    
    
def read_csvs(data_path, version=None):
    version = '_' if version is None else version
    return [pd.read_csv(p) for p in data_path.glob(f'rank_*.csv') if len(p.stem.split(version)) == 2]


err_fnc = {
    'weight>0': lambda x, y: min_error(x, y),
    'weight>=0': lambda x, y: min_error(x, y),
    'weight==0': lambda x, y: error_norm(1, x, y),
    'weight<0': lambda x, y: min_error(x, y),
    'weight<=0': lambda x, y: min_error(x, y)
}


def roc_auc_score(df, y, threshold=0, weight=None):
    import sklearn
    import sklearn.metrics as sm
    y_true = abs(df.weight) > threshold
    sample_weight = sklearn.utils.class_weight.compute_sample_weight('balanced', y_true)
    y_score = df[y].values if weight is None else df[y].values * weight
    return sm.roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        

def bootstrap_ci(bs_replicates, alpha=0.05):
    low, high = np.percentile(bs_replicates, [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
    return low, high


def bootstrap_pvalue(case, control, obs_diff, statistic=np.mean, alpha_ci=0.05):
    diffs = case - control
    low, high = np.percentile(diffs, [(alpha_ci / 2.0) * 100, (1 - alpha_ci / 2.0) * 100])

    diffs_shifted = diffs - statistic(diffs)

    emp_diff_pctile_rnk = scipy.stats.percentileofscore(diffs_shifted, obs_diff)

    auc_left = emp_diff_pctile_rnk / 100
    auc_right = 1 - auc_left
    pval = min([auc_left, auc_right]) * 2

    return pval, low, high, diffs, obs_diff


def rectify_keys(df, keys):
    result = df.copy()
    for key in keys:
        result[key] = df.apply(lambda x: x[key] if x[key] > 0 else 0, axis=1)
    return result


def compute_errors(
        data_path, version=None, rectify=False, sample=False, force_sample=False, 
        target_weights=['weight>=0'], threshold=0, keys=None, naive=True, compute_rsquared=False):
    keys = default_keys if keys is None else keys
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    errors = {target_weight: pd.DataFrame({'path': paths}) for target_weight in target_weights}
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        ranks = read_csvs(path, version=version)
        if len(ranks) == 0:
            continue
        ranksum = reduce_sum(ranks)
        ranksum = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for _, row in ranksum.iterrows()])
        
        with open(path / 'params.yaml', 'r') as f:
            params = yaml.load(f)
            
        
        if threshold:
            ranksum.loc[abs(ranksum.weight) < threshold, 'weight'] = 0
            
        if naive:
            ranksum = ranksum.merge(pd.read_csv(path / 'naive_cch.csv')[['pair', 'naive_cch']], on='pair')
        
        if rectify:
            ranksum = rectify_keys(ranksum, keys)
        params['exclude_path_stem'] = str(params['exclude_path_stem'])
        for target_weight in target_weights:
            errors[target_weight].loc[i, params.keys()] = params.values()
            
            df = ranksum.query(target_weight)
            if sample:
                try:
                    df = df.sample(sample)
                except Exception as e:
                    if force_sample:
                        continue
                    else:
                        raise e
                        
            for key in keys:          
                errors[target_weight].loc[i, 'error_' + key] = err_fnc[target_weight](df, key)
                if compute_rsquared:
                    errors[target_weight].loc[i, 'rsquared_' + key] = rsquared(df, key)
    return errors


def compute_error_trials(data_path, n_iter=100, n_samples=150, version=None, target_weights=['weight>=0'], threshold=0, keys=None, naive=True):
    keys = default_keys if keys is None else keys
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_dict = {target_weight: {i: defaultdict(list) for i in range(len(paths))} for target_weight in target_weights}
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter)
    for i, path in enumerate(paths):
        ranks = read_csvs(path, version=version)
        for ii in range(n_iter):
            idxs = rng.choice(len(ranks), size=n_samples, replace=False)
            ranksum = reduce_sum([ranks[j] for j in idxs])
            ranksum = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in ranksum.iterrows()])
            if threshold:
                ranksum.loc[abs(ranksum.weight) < threshold, 'weight'] = 0
            if naive:
                ranksum = ranksum.merge(pd.read_csv(path / 'naive_cch.csv')[['pair', 'naive_cch']], on='pair')
            for target_weight in target_weights:
                df = ranksum.query(target_weight)
                for key in keys:
                    data_dict[target_weight][i]['error_' + key].append(err_fnc[target_weight](df, key))
            pbar.update(1)
    pbar.close()

    data_dict = {k0: {k1: {k2: np.array(v2)
        for k2, v2 in v1.items()}
        for k1, v1 in v0.items()}
        for k0, v0 in data_dict.items()}
    return data_dict


def compute_error_confidence(errors, error_trials):
    for k in errors:
        for i, row in errors[k].iterrows():
            with open(row.path / 'params.yaml', 'r') as f:
                params = yaml.load(f)
            errors[k].loc[i, params.keys()] = params.values()

            statistic, pval = scipy.stats.wilcoxon(
                error_trials[k][i]['error_beta_ols_did'],
                error_trials[k][i]['error_beta_iv_did'],
            )
            errors[k].loc[i, 'error_ols_iv_did_pval'] = pval
            errors[k].loc[i, 'error_ols_iv_did_statistic'] = statistic

            error_trials[k][i]['error_diff_ols_iv_did'] = \
                error_trials[k][i]['error_beta_ols_did'] - error_trials[k][i]['error_beta_iv_did']
            error_trials[k][i]['error_diff_ols_iv'] = \
                error_trials[k][i]['error_beta_ols'] - error_trials[k][i]['error_beta_iv']

            errors[k].loc[i, 'error_diff_ols_iv_did'] = \
                np.mean(error_trials[k][i]['error_diff_ols_iv_did'])
            errors[k].loc[i, 'error_diff_ols_iv'] = \
                np.mean(error_trials[k][i]['error_diff_ols_iv'])

            errors[k].loc[i, 'error_diff_ols_iv_did_ci'] = \
                str(bootstrap_ci(error_trials[k][i]['error_diff_ols_iv_did']))
            errors[k].loc[i, 'error_diff_ols_iv_ci'] = \
                str(bootstrap_ci(error_trials[k][i]['error_diff_ols_iv']))
    return errors


def compute_error_convergence(data_path, version=None, target_weights=['weight>=0'], threshold=0, keys=None):
    keys = default_keys if keys is None else keys
    keys = [k for k in keys if not k.startswith('naive')]
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    n_samples = len(list(paths[0].glob('rank*.csv')))
    convergence = {t: {i: defaultdict(partial(np.empty, n_samples)) for i in range(len(paths))} for t in target_weights}
    pbar = tqdm(total=len(paths)*n_samples)
    for i, path in enumerate(paths):
        ranks = read_csvs(path, version=version)
        if len(ranks) == 0:
            print(f'Warning: zero length csv in {path}')
            continue
        for ll in range(n_samples):
            ranksum = reduce_sum(ranks[:ll+1])
            ranksum = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in ranksum.iterrows()])
            if threshold:
                ranksum.loc[abs(ranksum.weight) < threshold, 'weight'] = 0
            for target_weight in target_weights:
                df = ranksum.query(target_weight)
                for key in keys:
                    convergence[target_weight][i]['error_' + key][ll] =  err_fnc[target_weight](df, key)
                convergence[target_weight][i]['n_trials'][ll] = df.n_trials.values[0]
            pbar.update(1)
    pbar.close()
    return convergence


def compute_error_convergence_trials(data_path, n_samples=150, n_iter=10, version=None, target_weights=['weight>=0'], threshold=0, keys=None):
    keys = default_keys if keys is None else keys
    keys = [k for k in keys if not k.startswith('naive')]
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    convergence = {t: {i: defaultdict(partial(np.empty, (n_iter, n_samples))) for i in range(len(paths))} for t in target_weights}
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter*n_samples)
    for i, path in enumerate(paths):
        ranks = read_csvs(path, version=version)
        for ii in range(n_iter):
            idxs = rng.choice(len(ranks), size=n_samples, replace=False)
            for ll in range(len(idxs)):
                ranksum = reduce_sum([ranks[j] for j in idxs[:ll+1]])
                ranksum = pd.DataFrame([
                    compute_connectivity_from_sum(row)
                    for i, row in ranksum.iterrows()])
                if threshold:
                    ranksum.loc[abs(ranksum.weight) < threshold, 'weight'] = 0
                for target_weight in target_weights:
                    df = ranksum.query(target_weight)
                    for key in keys:
                        convergence[target_weight][i]['error_' + key][ii,ll] = err_fnc[target_weight](df, key)
                    convergence[target_weight][i]['n_trials'][ii, ll] = df.n_trials.values[0]
                pbar.update(1)
    pbar.close()
    return convergence


def compute_all_samples(data_path, version=None, naive=True):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    ranksums = {}
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        rank = read_csvs(path, version=version)
        if len(rank) == 0:
            continue
        ranksum = reduce_sum(rank)
        ranksums[i] = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for _, row in ranksum.iterrows()])
        if naive:
            ranksums[i] = ranksums[i].merge(pd.read_csv(path / 'naive_cch.csv')[['pair', 'naive_cch']], on='pair')
    return ranksums


def plot_errors(errors, y, keys=None, save=None, xlabel=None, measure='error', legend_kws={}, ylim=None):
    keys = default_keys if keys is None else keys
    errors = {k: df.sort_values(y) for k, df in errors.items()}
    label = lambda x: ','.join([labels[l] for l in x.split('_')[1:]])
    for target_weight, df in errors.items():
        fig, ax = plt.subplots(1,1)
        for key in keys:
            ax.plot(
                df[y], df[measure + '_' + key],
                label=fr'$\hat{{\beta}}_{{{label(key)}}}$',
                color=colors[label(key).lower()]
            )

        plt.legend(frameon=False, **legend_kws)
        if ylim is not None:
            plt.ylim(*ylim)
            plt.margins(y=0.1)
        sns.despine()
        ax.set_xlabel(y.capitalize() if xlabel is None else xlabel)
        measure_label = '\mathrm{Error}' if measure=='error' else r'\mathrm{R}^2'
        ax.set_ylabel(r'$\mathrm{Error}(w > 0)$' if target_weight=='weight>0'
            else fr'${{{measure_label}}}(w = 0)$' if target_weight=='weight==0'
            else fr'${{{measure_label}}}(w \geq 0)$' if target_weight=='weight>=0'
            else fr'${{{measure_label}}}(w \leq 0)$' if target_weight=='weight<=0'
            else fr'${{{measure_label}}}(w < 0)$')
        if save is not None:
            savefig(f'{save}_{target_weight}')
        
        
def plot_error_trials(error_trials, y, keys=None, alpha=0.5):
    keys = default_keys if keys is None else keys
    from matplotlib.lines import Line2D
    label = lambda x: ','.join([labels[l] for l in x.split('_')[1:]])
    for k, df in error_trials.items():
        fig, ax = plt.subplots(1,1)
        for key in keys:
            steps = df[y]
            errors = copy.deepcopy(df)
#             errors[key][errors[key]==0] = np.nan
            color = colors[label(key).lower()]
            ax.plot(steps.T, errors['error_' + key].T, color=color, alpha=alpha)
            ax.plot(steps[0], np.nanmean(errors['error_' + key], axis=0), color=color, label=fr'$\beta_{{{label(key)}}}$')
            if legend:
                ax.legend(frameon=False)
        ax.set_xscale('log')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{Error}(\beta)$')
        plt.title(k)


def plot_error_difference(errors, key):
    errors = {k: df.sort_values(key) for k, df in errors.items()}
    for k, df in errors.items():
        fig, ax = plt.subplots(1,1)
        ax.plot(df[key], df['error_diff_ols_iv_did'],
            label=r'$\mathrm{E}(\hat{\beta}_{OLS,DiD}) - \mathrm{E}(\hat{\beta}_{IV,DiD})$',
            color='C1')
        ax.fill_between(df[key],
            *np.array([eval(a) for a in df['error_diff_ols_iv_did_ci'].values]).T, alpha=.5,
            color='C1')

        ax.plot(df[key], df['error_diff_ols_iv'],
            label=r'$\mathrm{E}(\hat{\beta}_{OLS}) - \mathrm{E}(\hat{\beta}_{IV})$',
            color='C2')
        ax.fill_between(df[key],
            *np.array([eval(a) for a in df['error_diff_ols_iv_ci'].values]).T,
            alpha=.5, color='C2')

        plt.legend(frameon=False)
        sns.despine()
        ax.set_xlabel(key.capitalize())
        ax.set_ylabel(r'$\mathrm{Error}$')
        plt.title(k)
        
        
import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """
 
    name = 'squareroot'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform_non_affine(self, a): 
            return np.array(a)**0.5
 
        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()
 
    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform(self, a):
            return np.array(a)**2
 
        def inverted(self):
            return SquareRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.SquareRootTransform()
    
mscale.register_scale(SquareRootScale)

def plot_error_convergence(convergence, index, keys=None):
    
    keys = default_keys if keys is None else keys
    label = lambda x: ','.join([labels[l] for l in x.split('_')[1:]])
    for k, df in convergence.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        steps = df[index]['n_trials']
        errors = df[index]
        for key in keys:
            color = colors[label(key).lower()]
            ax.plot(steps, errors['error_' + key], color=color, label=fr'$\beta_{{{label(key)}}}$')

        plt.legend(frameon=False)
        ax.set_xscale('log')
#         ax.set_xscale('squareroot')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{error}(\beta)$')
        plt.title(k)


def plot_error_convergence_trials(convergence_trials, index, keys=None, alpha=0.2, axs=None, legend=True, xlabels=[True, True], ylabels=[True, True]):
    keys = default_keys if keys is None else keys
    import copy
    if axs is None:
        fig, axs = plt.subplots(len(convergence_trials),1, figsize=(5,5), dpi=150, sharex=True)
    label = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    for i, (target_weight, df) in enumerate(convergence_trials.items()):
        ax = axs[i]
        steps_t = df[index]['n_trials']
        errors_t = copy.deepcopy(df[index])
        for key in keys:
            key = key if key.startswith('error_') else 'error_' + key
            errors_t[key][errors_t[key]==0] = np.nan
            color = colors[label(key).lower()]
            ax.plot(steps_t.T, errors_t[key].T, color=color, alpha=alpha)
            ax.plot(steps_t[0], np.nanmean(errors_t[key], axis=0), color=color, label=fr'$\beta_{{{label(key)}}}$')
            if legend:
                ax.legend(frameon=False)
        ax.set_xscale('log')
#         ax.set_xscale('squareroot')
        sns.despine()
        if xlabels[i]:
            ax.set_xlabel('Trials')
        if ylabels[i]:
            ax.set_ylabel(r'$\mathrm{Error}(w > 0)$' if target_weight=='weight>0' 
                else r'$\mathrm{Error}(w = 0)$' if target_weight=='weight==0'
                else r'$\mathrm{Error}(w \geq 0)$' if target_weight=='weight>=0'
                else r'$\mathrm{Error}(w \leq 0)$' if target_weight=='weight<=0' 
                else r'$\mathrm{Error}(w < 0)$')


def plot_regression(df, keys=None, legend=True, rectify=False, scatter_color='hit_rate', fit_intercept=True, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    keys = default_keys if keys is None else keys
    df = rectify_keys(df, keys) if rectify else df
    
    fig, axs = plt.subplots(
        1, len(keys), figsize=(2*len(keys),2.5), dpi=150,
        sharey=True, sharex=True)
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for i, (ax, key) in enumerate(zip(axs, keys)):
        model = regplot(
            'weight', key, data=df,
            scatter_color=df[scatter_color], 
            colorbar=True, cax=cax, ax=ax, 
            fit_intercept=fit_intercept, **kwargs)
        if legend:
            h = plt.Line2D([], [], label='$R^2 = {:.3f}$'.format(model.rsquared), ls='-', color='k')
            ax.legend(handles=[h], frameon=False)
        if i == 0:
            ax.set_ylabel(r'$\hat{\beta}$')
        else:
            ax.set_ylabel('')
        sns.despine()
        ks = key.split('_')
        ax.set_title(fr'$\beta_{{{",".join(ks[1:])}}}$')


def plot_false_positives(df_zero, keys=None, scatter_kws=dict(s=5), violin_kws=dict(bw_method=0.5), clabel=None, rectify=False, scatter_color='hit_rate'):
    keys = default_keys if keys is None else keys
    df_zero = rectify_keys(df_zero, keys) if rectify else df_zero
    fig, ax = plt.subplots(1, 1, figsize=(4,2.5), dpi=150)
    pos = np.random.uniform(.25,.75, size=len(df_zero))
    for i, key in enumerate(keys):
        sc = ax.scatter(pos + .5 + i, df_zero[key], c=df_zero[scatter_color], **scatter_kws)

    cb = plt.colorbar(mappable=sc, ax=ax)
    cb.ax.yaxis.set_ticks_position('right')
    if clabel is not None:
        cb.set_label(clabel)

    violins = plt.violinplot(df_zero.loc[:, keys], showextrema=False, **violin_kws)
    for pc in violins['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('k')
        pc.set_alpha(0.6)
    plt.xticks(np.arange(len(keys))+1, [fr'$\beta_{{{",".join(key.split("_")[1:])}}}$' for key in keys])

    
def violin_compare_all(varlist, target_weight, errors, save=None):
    plt.figure(figsize=(2.5,3.5))
    key = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    viodf = pd.DataFrame()
    for var in varlist:
        v = pd.DataFrame()
        v['Error'] = errors[target_weight].loc[:, var]
        v[''] = key(var)
        viodf = pd.concat([viodf,v])
    sns.violinplot(
        data=viodf, x='', y='Error', inner="quart", linewidth=1, cut=0,
        palette={key(v): colors[key(v).lower()] for v in varlist}
    )
    ax = plt.gca()
    plt.ylabel(r'$\mathrm{Error}(w > 0)$' if target_weight=='weight>0' 
        else r'$\mathrm{Error}(w = 0)$' if target_weight=='weight==0'
        else r'$\mathrm{Error}(w \geq 0)$' if target_weight=='weight>=0'
        else r'$\mathrm{Error}(w \leq 0)$' if target_weight=='weight<=0' 
        else r'$\mathrm{Error}(w < 0)$')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    sns.despine()
    
    data_max = np.max(errors[target_weight].loc[:, varlist].values.max())
    data_min = np.min(errors[target_weight].loc[:, varlist].values.min())
    compare = [(v1,v2,x1,x2) for x1, v1 in enumerate(varlist) 
               for x2, v2 in zip(range(x1+1, len(varlist)), varlist[x1+1:])]
    
    stats = pd.DataFrame()
    for v1, v2, x1, x2 in compare:
        statistic, pvalue = scipy.stats.mannwhitneyu(
                    errors[target_weight].loc[:, v1],
                    errors[target_weight].loc[:, v2])
        stats.loc['_vs_'.join((v1,v2)), 'Statistic'] = statistic
        stats.loc['_vs_'.join((v1,v2)), 'Pvalue'] = pvalue
        # significance
        if pvalue < 0.0001:
            significance = "****"
        elif pvalue < 0.001:
            significance = "***"
        elif pvalue < 0.01:
            significance = "**"
        elif pvalue < 0.05:
            significance = "*"
        else:
            significance = "ns"
        d = x1

        y = (data_max * 1.05)
        h = 0.025 * (data_max - data_min)
        d_ =  d * 0.17 * (data_max - data_min)
        plt.plot([x1, x1, x2, x2], np.array([y - h, y, y, y - h]) + d_, c='k')
        plt.text(x2-.5, y + h + d_, significance, ha='center', va='bottom')
    if save is not None:
        savefig(f'{save}_{target_weight}')
        stats.to_csv(f'../paper/graphics/{save}_{target_weight}.csv')


def load(fn):
    data = np.load(fn, allow_pickle=True)
    data = {k: data[k][()] for k in data.keys()}
    X = data['data']
    W_0 = data['W_0']
    W = data['W']
    params = data['params']
    return X, W_0, W, params


def compute_condition(data_path, lag=1):
    from causal_optoconnectics.tools import roll_pad, decompress_events
    from scipy.linalg import svd
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_df = pd.DataFrame({'path': paths})

    values = pd.DataFrame()
    iterator = tqdm(data_df.iterrows(), total=len(data_df))
    for i, row in iterator:
        iterator.set_description(row.path.stem)

        X, W_0, W, params = load(row.path / 'rank_0.npz')

        data_df.loc[i, params.keys()] = params.values()
        if 'glorot_normal' in params:
            data_df.loc[i, 'sigma'] = params['glorot_normal']['sigma']

        x = decompress_events(X, len(W), params['n_time_step'])
        
        cov_x = np.cov(x[:len(W_0)], roll_pad(x[:len(W_0)], lag))
        s_cov = svd(cov_x, compute_uv=False)
        data_df.loc[i, 'cov_condition'] = s_cov.max() / s_cov.min()
        data_df.loc[i, 'cov_smin'] = s_cov.min()
        data_df.loc[i, 'cov_smax'] = s_cov.max()
    return data_df
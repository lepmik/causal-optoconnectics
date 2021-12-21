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
    compute_connectivity_from_sum
)
from causal_optoconnectics.core import Connectivity

colors = {
    'iv,did': '#e41a1c',
    'iv': '#e41a1c',
    'brew,did': 'C2',
    'brew': 'C2',
    'br,did': 'C2',
    'br': 'C2',
    'ols,did': '#377eb8',
    'ols': '#377eb8'
}

labels = {'did': 'DiD', 'ols': 'OLS', 'iv':'IV', 'brew': 'BR'}

keys = [
    'beta_ols', 'beta_iv', 'beta_brew', 
    'beta_ols_did', 'beta_iv_did', 'beta_brew_did']


def savefig(stem):
    fname = pathlib.Path(f'../paper/graphics/{stem}').with_suffix('.svg')
    plt.savefig(fname, bbox_inches='tight', transparent=True)
    
    
def read_csvs(data_path, version=None):
    version = '_' if version is None else version
    return [pd.read_csv(p) for p in data_path.glob(f'rank_*.csv') if len(p.stem.split(version)) == 2]


err_fnc = {
    'weight>0': lambda x, y: min_error(x, y).fun,
    'weight==0': lambda x, y: error_norm(1, x, y),
    'weight<0': lambda x, y: min_error(x, y).fun
}


def roc_auc_score(df, y):
        import sklearn
        import sklearn.metrics as sm
        y_true = df.weight > 0
        sample_weight = sklearn.utils.class_weight.compute_sample_weight('balanced', y_true)
        y_score = df[y].values
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


def compute_errors(data_path, version=None, rectify=False, sample=False, force_sample=False, target_weights=['weight>0', 'weight==0']):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    errors = {target_weight: pd.DataFrame({'path': paths}) for target_weight in target_weights}
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        ranks = read_csvs(path, version=version)
        if len(ranks) == 0:
            continue
        ranksum = reduce_sum(ranks)
        ranksum = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for i, row in ranksum.iterrows()])
        
        with open(path / 'params.yaml', 'r') as f:
            params = yaml.load(f)
            
        ranksum = rectify_keys(ranksum, keys) if rectify else ranksum
        for target_weight in target_weights:
            df = ranksum.query(target_weight)
            if sample:
                try:
                    df = df.sample(sample)
                except Exception as e:
                    if force_sample:
                        continue
                    else:
                        raise e
            errors[target_weight].loc[i, params.keys()] = params.values()  
            for key in keys:          
                errors[target_weight].loc[i, 'error_' + key] = err_fnc[target_weight](df, key)
    return errors


def compute_error_trials(data_path, n_iter=100, n_samples=150, version=None, target_weights=['weight>0', 'weight==0']):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_dict = {target_weight: {i: defaultdict(list) for i in range(len(paths))} for target_weight in target_weights}
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter)
    for i, path in enumerate(paths):
        samples = read_csvs(path, version=version)
        for ii in range(n_iter):
            idxs = rng.choice(len(samples), size=n_samples, replace=False)
            sample = reduce_sum([samples[j] for j in idxs])
            sample = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in sample.iterrows()])
            for target_weight in target_weights:
                df = sample.query(target_weight)
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


def compute_error_convergence(data_path, version=None, target_weights=['weight>0', 'weight==0']):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    n_samples = len(list(paths[0].glob('rank*.csv')))
    convergence = {t: {i: defaultdict(partial(np.empty, n_samples)) for i in range(len(paths))} for t in target_weights}
    pbar = tqdm(total=len(paths)*n_samples)
    for i, path in enumerate(paths):
        samples = read_csvs(path, version=version)
        if len(samples) == 0:
            continue
        for ll in range(n_samples):
            sample = reduce_sum(samples[:ll+1])
            sample = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in sample.iterrows()]).dropna()
            for target_weight in target_weights:
                df = sample.query(target_weight)
                for key in keys:
                    convergence[target_weight][i]['error_' + key][ll] =  err_fnc[target_weight](df, key)
                convergence[target_weight][i]['n_trials'][ll] = np.nan if len(sample)==0 else sample.n_trials.values[0]
            pbar.update(1)
    pbar.close()
    return convergence


def compute_error_convergence_trials(data_path, n_samples=150, n_iter=10, version=None, target_weights=['weight>0', 'weight==0']):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    convergence = {t: {i: defaultdict(partial(np.empty, (n_iter, n_samples))) for i in range(len(paths))} for t in target_weights}
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter*n_samples)
    for i, path in enumerate(paths):
        samples = read_csvs(path, version=version)
        for ii in range(n_iter):
            idxs = rng.choice(len(samples), size=n_samples, replace=False)
            for ll in range(len(idxs)):
                sample = reduce_sum([samples[j] for j in idxs[:ll+1]])
                sample = pd.DataFrame([
                    compute_connectivity_from_sum(row)
                    for i, row in sample.iterrows()]).dropna()
                for target_weight in target_weights:
                    df = sample.query(target_weight)
                    for key in keys:
                        convergence[target_weight][i]['error_' + key][ii,ll] = err_fnc[target_weight](df, key)
                    convergence[target_weight][i]['n_trials'][ii, ll] = np.nan if len(df)==0 else df.n_trials.values[0]
                pbar.update(1)
    pbar.close()
    return convergence


def compute_all_samples(data_path, version=None):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    grand_samples = {}
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        samples = read_csvs(path, version=version)
        if len(samples) == 0:
            continue
        sample = reduce_sum(samples)
        grand_samples[i] = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for i, row in sample.iterrows()])
    return grand_samples


def plot_errors(errors, y):
    errors = {k: df.sort_values(y) for k, df in errors.items()}
    label = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    for target_weight, df in errors.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        for key in keys:
            ax.plot(df[y], df['error_' + key],
                label=fr'$\hat{{\beta}}_{{{label("error_" + key)}}}$')

        plt.legend(frameon=False)
        sns.despine()
        ax.set_xlabel(y.capitalize())
        ax.set_ylabel(r'$\mathrm{Error}$')
        plt.title(target_weight)
        
        
def plot_error_trials(error_trials, keys, alpha=0.5):
    from matplotlib.lines import Line2D
    label = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    for k, df in error_trials.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        for key in keys:
            steps = df[key]
            errors = copy.deepcopy(df[key])
#             errors[key][errors[key]==0] = np.nan
            color = colors[label(key).lower()]
            ax.plot(steps.T, errors[key].T, color=color, alpha=alpha)
            ax.plot(steps[0], np.nanmean(errors[key], axis=0), color=color, label=fr'$\beta_{{{label(key)}}}$')
            if legend:
                ax.legend(frameon=False)
        ax.set_xscale('log')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{error}(\beta)$')
        plt.title(k)


def plot_error_difference(errors, key):
    errors = {k: df.sort_values(key) for k, df in errors.items()}
    for k, df in errors.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
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


def plot_error_convergence(convergence, index, keys):
    label = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    for k, df in convergence.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        steps = df[index]['n_trials']
        errors = df[index]
        for key in keys:
            color = colors[label(key).lower()]
            ax.plot(steps, errors[key], color=color, label=fr'$\beta_{{{label(key)}}}$')

        plt.legend(frameon=False)
        ax.set_xscale('log')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{error}(\beta)$')
        plt.title(k)


def plot_error_convergence_trials(convergence_trials, index, keys, alpha=0.2, axs=None, legend=True, xlabels=[True, True], ylabels=[True, True]):
    import copy
    if axs is None:
        fig, axs = plt.subplots(2,1, figsize=(5,5), dpi=150, sharex=True)
    label = lambda x: ','.join([labels[l] for l in x.split('_')[2:]])
    for i, (k, df) in enumerate(convergence_trials.items()):
        ax = axs[i]
        steps_t = df[index]['n_trials']
        errors_t = copy.deepcopy(df[index])
        for key in keys:
            errors_t[key][errors_t[key]==0] = np.nan
            color = colors[label(key).lower()]
            ax.plot(steps_t.T, errors_t[key].T, color=color, alpha=alpha)
            ax.plot(steps_t[0], np.nanmean(errors_t[key], axis=0), color=color, label=fr'$\beta_{{{label(key)}}}$')
            if legend:
                ax.legend(frameon=False)
        ax.set_xscale('log')
        sns.despine()
        if xlabels[i]:
            ax.set_xlabel('Trials')
        if ylabels[i]:
            ax.set_ylabel(r'$Error(w > 0)$' if k=='positives' else fr'$Error(w = 0)$' if k=='zeroes' else fr'$Error(w < 0)$')


def plot_regression(df, keys=['beta_ols_did','beta_iv_did','beta_brew_did'], legend=True, rectify=False, **kwargs):
    df = rectify_keys(df, keys) if rectify else df
    fig, axs = plt.subplots(1,len(keys),figsize=(4,2.5), dpi=150, sharey=True, sharex=True)
    for i, (ax, key) in enumerate(zip(axs, keys)):
        model = regplot(
            'weight', key, data=df,
            scatter_color=df['hit_rate'], colorbar=i==len(keys)-1, ax=ax, **kwargs)
        if legend:
            h = plt.Line2D([], [], label='$R^2 = {:.3f}$'.format(model.rsquared), ls='-', color='k')
            ax.legend(handles=[h], frameon=False)
        if i == 0:
            ax.set_ylabel(r'$\hat{\beta}$')
        else:
            ax.set_ylabel('')
        sns.despine()
        ks = key.split('_')
        ax.set_title(fr'$\{ks[0]}_{{{",".join(ks[1:])}}}$')


def plot_false_positives(df_zero, keys=['beta_ols_did', 'beta_iv_did', 'beta_brew_did'], scatter_kws=dict(s=5), violin_kws=dict(bw_method=0.5), clabel=None, rectify=False):
    df_zero = rectify_keys(df_zero, keys) if rectify else df_zero
    fig, ax = plt.subplots(1, 1, figsize=(4,2.5), dpi=150)
    pos = np.random.uniform(.25,.75, size=len(df_zero))
    for i, key in enumerate(keys):
        sc = ax.scatter(pos + .5 + i, df_zero[key], c=df_zero.hit_rate, **scatter_kws)

    cb = plt.colorbar(mappable=sc, ax=ax)
    cb.ax.yaxis.set_ticks_position('right')
    if clabel is not None:
        cb.set_label(clabel)

    violins = plt.violinplot(df_zero.loc[:, keys], showextrema=False, **violin_kws)
    for pc in violins['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('k')
        pc.set_alpha(0.6)
    plt.xticks(np.arange(len(keys))+1, [fr'$\{key.split("_")[0]}_{{{",".join(key.split("_")[1:])}}}$' for key in keys])

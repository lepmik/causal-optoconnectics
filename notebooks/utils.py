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


err_fnc = {
    'positives': lambda x, y: min_error(x, y).fun,
    'negatives': lambda x, y: error_norm(1, x, y)
}

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


def compute_error_trials(data_path, n_iter = 100, n_samples = 150):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_dict = {
        'positives': {i: defaultdict(list) for i in range(len(paths))},
        'negatives': {i: defaultdict(list) for i in range(len(paths))}
    }
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter)
    for i, path in enumerate(paths):
        samples = [pd.read_csv(p) for p in path.glob('rank*.csv')]
        for ii in range(n_iter):
            idxs = rng.choice(len(samples), size=n_samples, replace=False)
            sample = reduce_sum([samples[j] for j in idxs])
            sample = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in sample.iterrows()])
            for k, q in zip(['positives', 'negatives'], ['weight>0', 'weight==0']):
                df = sample.query(q)
                data_dict[k][i]['error_beta_ols_did'].append(err_fnc[k](df, 'beta_ols_did'))
                data_dict[k][i]['error_beta_iv_did'].append(err_fnc[k](df, 'beta_iv_did'))
                data_dict[k][i]['error_beta_brew_did'].append(err_fnc[k](df, 'beta_brew_did'))
                data_dict[k][i]['error_beta_ols'].append(err_fnc[k](df, 'beta_ols'))
                data_dict[k][i]['error_beta_iv'].append(err_fnc[k](df, 'beta_iv'))
                data_dict[k][i]['error_beta_brew'].append(err_fnc[k](df, 'beta_brew'))
            pbar.update(1)
    pbar.close()

    data_dict = {k0: {k1: {k2: np.array(v2)
        for k2, v2 in v1.items()}
        for k1, v1 in v0.items()}
        for k0, v0 in data_dict.items()}
    return data_dict


def compute_errors(data_path):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    errors = {
        'positives': pd.DataFrame({'path': paths}),
        'negatives': pd.DataFrame({'path': paths})
    }
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        samples = [pd.read_csv(p) for p in path.glob('rank*.csv')]
        sample = reduce_sum([samples[j] for j in idxs])
        sample = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for i, row in sample.iterrows()])

        with open(path / 'params.yaml', 'r') as f:
            params = yaml.load(f)
        for k, q in zip(['positives', 'negatives'], ['weight>0', 'weight==0']):
            df = sample.query(q)
            errors[k].loc[i, params.keys()] = params.values()
            errors[k].loc[i, 'error_beta_ols_did'] = err_fnc[k](df, 'beta_ols_did')
            errors[k].loc[i, 'error_beta_iv_did'] = err_fnc[k](df, 'beta_iv_did')
            errors[k].loc[i, 'error_beta_brew_did'] = err_fnc[k](df, 'beta_brew_did')
            errors[k].loc[i, 'error_beta_ols'] = err_fnc[k](df, 'beta_ols')
            errors[k].loc[i, 'error_beta_iv'] = err_fnc[k](df, 'beta_iv')
            errors[k].loc[i, 'error_beta_brew'] = err_fnc[k](df, 'beta_brew')
    return errors


def compute_error_convergence(data_path):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    n_samples = len(list(paths[0].glob('rank*.csv')))
    convergence = {
        'positives': {i: defaultdict(partial(np.empty, n_samples)) for i in range(len(paths))},
        'negatives': {i: defaultdict(partial(np.empty, n_samples)) for i in range(len(paths))}
    }
    pbar = tqdm(total=len(paths)*n_samples)
    for i, path in enumerate(paths):
        samples = [pd.read_csv(p) for p in path.glob('rank*.csv')]
        for ll in range(n_samples):
            sample = reduce_sum(samples[:ll+1])
            sample = pd.DataFrame([
                compute_connectivity_from_sum(row)
                for i, row in sample.iterrows()]).dropna()
            for k, q in zip(['positives', 'negatives'], ['weight>0', 'weight==0']):
                df = sample.query(q)
                convergence[k][i]['error_beta_ols_did'][ll] =  err_fnc[k](df, 'beta_ols_did')
                convergence[k][i]['error_beta_iv_did'][ll] = err_fnc[k](df, 'beta_iv_did')
                convergence[k][i]['error_beta_brew_did'][ll] = err_fnc[k](df, 'beta_brew_did')
                convergence[k][i]['error_beta_ols'][ll] = err_fnc[k](df, 'beta_ols')
                convergence[k][i]['error_beta_iv'][ll] = err_fnc[k](df, 'beta_iv')
                convergence[k][i]['error_beta_brew'][ll] = err_fnc[k](df, 'beta_brew')
                convergence[k][i]['n_trials'][ll] = np.nan if len(sample)==0 else sample.n_trials.values[0]
            pbar.update(1)
    pbar.close()
    return convergence


def compute_error_convergence_trials(data_path, n_samples=150, n_iter=10):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    convergence = {
        'positives': {i: defaultdict(partial(np.empty, (n_iter, n_samples))) for i in range(len(paths))},
        'negatives': {i: defaultdict(partial(np.empty, (n_iter, n_samples))) for i in range(len(paths))}
    }
    rng = default_rng()
    pbar = tqdm(total=len(paths)*n_iter*n_samples)
    for i, path in enumerate(paths):
        samples = [pd.read_csv(p) for p in path.glob('rank*.csv')]
        for ii in range(n_iter):
            idxs = rng.choice(len(samples), size=n_samples, replace=False)
            for ll in range(len(idxs)):
                sample = reduce_sum([samples[j] for j in idxs[:ll+1]])
                sample = pd.DataFrame([
                    compute_connectivity_from_sum(row)
                    for i, row in sample.iterrows()]).dropna()
                for k, q in zip(['positives', 'negatives'], ['weight>0', 'weight==0']):
                    df = sample.query(q)
                    convergence[k][i]['error_beta_ols_did'][ii,ll] = err_fnc[k](df, 'beta_ols_did')
                    convergence[k][i]['error_beta_iv_did'][ii,ll] = err_fnc[k](df, 'beta_iv_did')
                    convergence[k][i]['error_beta_brew_did'][ii,ll] = err_fnc[k](df, 'beta_brew_did')
                    convergence[k][i]['error_beta_ols'][ii,ll] = err_fnc[k](df, 'beta_ols')
                    convergence[k][i]['error_beta_iv'][ii,ll] = err_fnc[k](df, 'beta_iv')
                    convergence[k][i]['error_beta_brew'][ii,ll] = err_fnc[k](df, 'beta_brew')
                    convergence[k][i]['n_trials'][ii, ll] = np.nan if len(df)==0 else df.n_trials.values[0]
                pbar.update(1)
    pbar.close()
    return convergence


def compute_all_samples(data_path):
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    grand_samples = {}
    for i, path in tqdm(enumerate(paths)):
        samples = [pd.read_csv(p) for p in path.glob('rank*.csv')]
        sample = reduce_sum(samples)
        grand_samples[i] = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for i, row in sample.iterrows()])
    return grand_samples


def compute_error_confidence(errors):
    for k in errors:
        for i, row in errors[k].iterrows():
            with open(row.path / 'params.yaml', 'r') as f:
                params = yaml.load(f)
            errors[k].loc[i, params.keys()] = params.values()

            statistic, pval = scipy.stats.wilcoxon(
                data_dict[k][i]['error_beta_ols_did'],
                data_dict[k][i]['error_beta_iv_did'],
            )
            errors[k].loc[i, 'error_ols_iv_did_pval'] = pval
            errors[k].loc[i, 'error_ols_iv_did_statistic'] = statistic

            data_dict[k][i]['error_diff_ols_iv_did'] = \
                data_dict[k][i]['error_beta_ols_did'] - data_dict[k][i]['error_beta_iv_did']
            data_dict[k][i]['error_diff_ols_iv'] = \
                data_dict[k][i]['error_beta_ols'] - data_dict[k][i]['error_beta_iv']

            errors[k].loc[i, 'error_diff_ols_iv_did'] = \
                np.mean(data_dict[k][i]['error_diff_ols_iv_did'])
            errors[k].loc[i, 'error_diff_ols_iv'] = \
                np.mean(data_dict[k][i]['error_diff_ols_iv'])

            errors[k].loc[i, 'error_diff_ols_iv_did_ci'] = \
                str(bootstrap_ci(data_dict[k][i]['error_diff_ols_iv_did']))
            errors[k].loc[i, 'error_diff_ols_iv_ci'] = \
                str(bootstrap_ci(data_dict[k][i]['error_diff_ols_iv']))
    return errors

def plot_error(errors, key):
    errors = {k: df.sort_values(key) for k, df in errors.items()}
    for k, df in errors.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        ax.plot(df[key], df['error_beta_ols_did'],
            label=r'$\hat{\beta}_{OLS,DiD}$', color='C0')

        ax.plot(df[key], df['error_beta_iv_did'],
            label=r'$\hat{\beta}_{IV,DiD}$', color='C1')

        ax.plot(df[key], df['error_beta_brew_did'],
            label=r'$\hat{\beta}_{BR,DiD}$', color='C2')

        ax.plot(df[key], df['error_beta_ols'],
            label=r'$\hat{\beta}_{OLS}$', color='C3')

        ax.plot(df[key], df['error_beta_iv'],
            label=r'$\hat{\beta}_{IV}$', color='C4')

        ax.plot(df[key], df['error_beta_brew'],
            label=r'$\hat{\beta}_{BR}$', color='C5')

        plt.legend(frameon=False)
        sns.despine()
        ax.set_xlabel(key.capitalize())
        ax.set_ylabel(r'$\mathrm{Error}$')
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


def plot_error_convergence(convergence, index):
    for k, df in convergence.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        steps = df[index]['n_trials']
        errors = df[index]

        ax.plot(steps, errors['error_beta_ols_did'], label=r'$\hat{\beta}_{OLS,DiD}$', color='C0')

        ax.plot(steps, errors['error_beta_iv_did'], label=r'$\hat{\beta}_{IV,DiD}$', color='C1')

        ax.plot(steps, errors['error_beta_brew_did'], label=r'$\hat{\beta}_{BR,DiD}$', color='C2')

        ax.plot(steps, errors['error_beta_ols'], label=r'$\hat{\beta}_{OLS}$', color='C3')

        ax.plot(steps, errors['error_beta_iv'], label=r'$\hat{\beta}_{IV}$', color='C4')

        ax.plot(steps, errors['error_beta_brew'], label=r'$\hat{\beta}_{BR}$', color='C5')

        plt.legend(frameon=False)
        ax.set_xscale('log')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{error}(\beta)$')
        plt.title(k)


def plot_error_convergence_trials(convergence, index, alpha=0.5):
    from matplotlib.lines import Line2D
    for k, df in convergence.items():
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=150)
        steps = df[index]['n_trials'].T
        errors = df[index]

        ax.plot(steps, errors['error_beta_ols_did'].T, color='C0', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_ols_did'].mean(), color='C0', alpha=alpha)

        ax.plot(steps, errors['error_beta_iv_did'].T, color='C1', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_iv_did'].mean(), color='C1', alpha=alpha)

        ax.plot(steps, errors['error_beta_brew_did'].T, color='C2', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_brew_did'].mean(), color='C2', alpha=alpha)

        ax.plot(steps, errors['error_beta_ols'].T, color='C3', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_ols'].mean(), color='C3', alpha=alpha)

        ax.plot(steps, errors['error_beta_iv'].T, color='C4', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_iv'].mean(), color='C4', alpha=alpha)

        ax.plot(steps, errors['error_beta_brew'].mean(), color='C5', alpha=alpha)
        ax.plot(steps[0], errors['error_beta_brew'].mean(), color='C5', alpha=alpha)

        plt.legend(
            handles=[Line2D([],[],c=c) for c in ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']],
            labels=[r'$\hat{\beta}_{OLS,DiD}$', r'$\hat{\beta}_{IV,DiD}$', r'$\hat{\beta}_{BR,DiD}$',
                    r'$\hat{\beta_{OLS}}$',r'$\hat{\beta}_{IV}$', r'$\hat{\beta}_{BR}$'],
            frameon=False)
        ax.set_xscale('log')
        sns.despine()
        ax.set_xlabel('Trials')
        ax.set_ylabel(r'$\mathrm{error}(\beta)$')
        plt.title(k)


def plot_regression(samples, index, keys=['beta_ols_did','beta_iv_did','beta_brew_did']):
    df = samples[index].query('weight>0')
    fig, axs = plt.subplots(1,3,figsize=(12,4), dpi=150, sharey=True)
    for i, (ax, key) in enumerate(zip(axs, keys)):
        model = regplot(
            'weight', key, data=df,
            scatter_color=df['hit_rate'], colorbar=True, ax=ax)

        h = plt.Line2D([], [], label='$R^2 = {:.3f}$'.format(model.rsquared), ls='-', color='k')
        ax.legend(handles=[h])
        if i == 0:
            ax.set_ylabel(r'$\hat{\beta}$')
        else:
            ax.set_ylabel('')
        sns.despine()
        ks = key.split('_')
        ax.set_title(fr'$\{ks[0]}_{{{",".join(ks[1:])}}}$')


def plot_false_positives(samples, index, keys=['beta_ols_did', 'beta_iv_did', 'beta_brew_did']):
    df_zero = samples[index].query('weight==0')
    fig, ax = plt.subplots(1, 1, figsize=(6,5), dpi=150)
    pos = np.random.uniform(.25,.75, size=len(df_zero))
    ax.scatter(pos + .5, df_zero[keys[0]], c=df_zero.hit_rate, s=5)
    sc = ax.scatter(pos + 1.5, df_zero[keys[0]], c=df_zero.hit_rate, s=5)
    ax.scatter(pos + 2.5, df_zero[keys[0]], c=df_zero.hit_rate, s=5)

    cb = plt.colorbar(mappable=sc, ax=ax)
    cb.ax.yaxis.set_ticks_position('right')

    violins = plt.violinplot(df_zero.loc[:, keys], showextrema=False, bw_method=0.5)
    for pc in violins['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('k')
        pc.set_alpha(0.6)
    plt.xticks([1, 2, 3], [fr'$\{key.split("_")[0]}_{{{",".join(key.split("_")[1:])}}}$' for key in keys])

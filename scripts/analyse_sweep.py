import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib
from collections import defaultdict
from scipy.linalg import svd
from functools import partial
import ruamel.yaml
from scipy.linalg import norm
from scipy.optimize import minimize_scalar

from causal_optoconnectics.graphics import regplot, scatterplot, probplot
from causal_optoconnectics.tools import conditional_probability, joint_probability, roll_pad
from causal_optoconnectics.tools import compute_trials_multi, decompress_events
from causal_optoconnectics.core import Connectivity


x_i, x_j = 11, 13
y_i, y_j = 12, 19
z_i, z_j = 7, 10

def process_metadata(W, stim_index, params):

    pairs = []
    for i in range(params['n_neurons']):
        for j in range(params['n_neurons']):
            if i==j:
                continue
            pair = f'{i}_{j}'
            pairs.append({
                'source': i,
                'target': j,
                'pair': pair,
                'weight': W[i, j, 0],
                'source_stim': W[stim_index, i, 0] > 0,
                'source_stim_strength': W[stim_index, i, 0],
                'target_stim': W[stim_index, j, 0] > 0,
            })
    return pd.DataFrame(pairs)

def process(pair, trials, W, stim_index, params, n_trials=None):
    i, j = [int(a) for a in pair.split('_')]

    pre, post = trials[i], trials[j]

    n_trials = len(pre) if n_trials is None else n_trials

    conn = Connectivity(pre[:n_trials], post[:n_trials], x_i, x_j, y_i, y_j, z_i, z_j)

    result ={
        'source': i,
        'target': j,
        'pair': pair,
        'beta_iv': conn.beta_iv,
        'beta': conn.beta,
        'beta_iv_did': conn.beta_iv_did,
        'beta_did': conn.beta_did,
        'hit_rate': conn.hit_rate,
        'weight': W[i, j, 0],
        'source_stim': W[stim_index, i, 0] > 0,
        'source_stim_strength': W[stim_index, i, 0],
        'target_stim': W[stim_index, j, 0] > 0,
    }
    result.update(params)
    return result


def compute_time_dependence(i, j, step=10000):
    pre, post = trials[i], trials[j]
    results = []
    start = 0
    for stop in tqdm(range(step, len(pre) + step, step)):
        results.append(process(i,j,stop))
    return results



def error(a, df, key):
    return df['weight'] - a * df[key]

def error_norm(a, df, key):
    return norm(error(a, df, key))

def min_error(df, key):
    return minimize_scalar(error_norm, args=(df, key)).fun

def load(path):
    data = np.load(path / 'rank_0.npz', allow_pickle=True)
    data = {k: data[k][()] for k in data.keys()}
    data['data'] = [np.load(fn, allow_pickle=True)['data'][()] for fn in path.glob('*.npz')]
    return data

if __name__ == '__main__':
    import sys
    yaml = ruamel.yaml.YAML()
    data_path = pathlib.Path(sys.argv[1]).absolute().resolve()
    print(f'Analyzing {data_path}')
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_df = pd.DataFrame({'path': paths})

    values = pd.DataFrame()
    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        #print(f'Working on {row.path}')
        data = load(row.path)
        X = data['data']
        W_0 = data['W_0']
        W = data['W']
        stim_index = len(W_0)
        params = data['params']
        data_df.loc[i, params.keys()] = params.values()
        data_df.loc[i, 'sigma'] = params['glorot_normal']['sigma']
        with open(row.path / 'params.yaml', 'w') as f:
            yaml.dump(params, f)
        n_neurons = params['n_neurons']
        trials = compute_trials_multi(X, len(W_0), stim_index)
        np.savez(row.path / 'trials', data=trials)
        
        s_W = svd(W_0, compute_uv=False)
        data_df.loc[i, 'W_condition'] = s_W.max() / s_W.min()
        data_df.loc[i, 'W_smin'] = s_W.min()
        data_df.loc[i, 'W_smax'] = s_W.max()

        x = decompress_events(X[0], len(W), params['n_time_step'])
        cov_x = np.cov(x[:len(W_0)])
        s_cov = svd(cov_x, compute_uv=False)
        data_df.loc[i, 'cov_condition'] = s_cov.max() / s_cov.min()
        data_df.loc[i, 'cov_smin'] = s_cov.min()
        data_df.loc[i, 'cov_smax'] = s_cov.max()

        results_meta = process_metadata(W=W, stim_index=stim_index, params=params)
        sample_meta = results_meta.query('source_stim and not target_stim and weight >= 0')
        sample = pd.DataFrame([process(pair=pair, W=W, stim_index=stim_index, trials=trials, params=params) for pair in sample_meta.pair.values])
        #sample = multi_process(trials=trials, W=W, stim_index=stim_index, params=params, pairs=sample_meta.pair.values)
        sample.to_csv(row.path / 'sample.csv')
        values = pd.concat((values, sample))
        data_df.loc[i, 'error_beta_did'] = min_error(sample, 'beta_did')
        data_df.loc[i, 'error_beta_iv_did'] = min_error(sample, 'beta_iv_did')
        data_df.loc[i, 'error_beta'] = min_error(sample, 'beta')
        data_df.loc[i, 'error_beta_iv'] = min_error(sample, 'beta_iv')

    data_df.loc[:,'error_diff'] = data_df.error_beta - data_df.error_beta_iv
    data_df.to_csv(data_path / 'summary.csv')
    values.to_csv(data_path / 'values.csv')

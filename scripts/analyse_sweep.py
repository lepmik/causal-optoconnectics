import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib
from collections import defaultdict
from scipy.linalg import svd
from functools import partial
import ruamel.yaml

from causal_optoconnectics.tools import (
    compute_trials_multi,
    decompress_events,
    error,
    error_norm,
    min_error,
    process,
    process_metadata
)

rparams = {
    'x1': 11,
    'x2': 13,
    'y1': 12,
    'y2': 19,
    'z1': 7,
    'z2': 10,
}

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
        params.update(rparams)
        data_df.loc[i, params.keys()] = params.values()
        data_df.loc[i, 'sigma'] = params['glorot_normal']['sigma']
        with open(row.path / 'params.yaml', 'w') as f:
            yaml.dump(params, f)
        n_neurons = params['n_neurons']

        results_meta = process_metadata(W=W, stim_index=stim_index, params=params)
        sample_meta = results_meta.query('source_stim and not target_stim and weight >= 0')
        neurons = pd.concat((sample_meta.source, sample_meta.target)).unique()
        trials = compute_trials_multi(X, neurons, stim_index)
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

        sample = pd.DataFrame([process(pair=pair, W=W, stim_index=stim_index, trials=trials, params=params) for pair in sample_meta.pair.values])
        #sample = multi_process(trials=trials, W=W, stim_index=stim_index, params=params, pairs=sample_meta.pair.values)
        sample.to_csv(row.path / 'sample.csv')
        values = pd.concat((values, sample))
        data_df.loc[i, 'error_beta_did'] = min_error(sample, 'beta_did').fun
        data_df.loc[i, 'error_beta_iv_did'] = min_error(sample, 'beta_iv_did').fun
        data_df.loc[i, 'error_beta'] = min_error(sample, 'beta').fun
        data_df.loc[i, 'error_beta_iv'] = min_error(sample, 'beta_iv').fun

    data_df.loc[:,'error_diff'] = data_df.error_beta - data_df.error_beta_iv
    data_df.loc[:,'error_diff_did'] = data_df.error_beta_did - data_df.error_beta_iv_did
    data_df.to_csv(data_path / 'summary.csv')
    values.to_csv(data_path / 'values.csv')

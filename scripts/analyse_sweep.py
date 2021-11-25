import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib
from collections import defaultdict
from scipy.linalg import svd
from functools import partial
import ruamel.yaml
import multiprocessing
from causal_optoconnectics.core import Connectivity
from causal_optoconnectics.tools import (
    compute_trials,
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

def reduce_sum(dfs):
    keys = [
        'yz_sum',
        'z_sum',
        'yx_sum',
        'x_sum',
        'yxinv_sum',
        'xinv_sum',
        'y0z_sum',
        'y0x_sum',
        'y0xinv_sum',
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
    conn = Connectivity(
        compute_values=False,
        compute_sums=False
    )
    conn.__dict__.update(row.to_dict())
    conn.compute()
    result = conn.__dict__
    return result


def load(fn):
    data = np.load(fn, allow_pickle=True)
    data = {k: data[k][()] for k in data.keys()}
    X = data['data']
    W_0 = data['W_0']
    W = data['W']
    params = data['params']
    params.update(rparams)
    return X, W_0, W, params


def compute(fn):
    X, W_0, W, params = load(fn)
    params.pop('seed')
    stim_index = len(W_0)
    results_meta = pd.DataFrame(process_metadata(range(len(W_0)), range(len(W_0)), W=W, stim_index=stim_index))
    sample_meta = results_meta.query('source_stim and not target_stim and weight >= 0')
    neurons = pd.concat((sample_meta.source, sample_meta.target)).unique()
    trials = compute_trials(X, neurons, stim_index)
    sums = pd.DataFrame([
        process(pair=pair, W=W, stim_index=stim_index, trials=trials, params=params, compute_values=False, compute_sums=True)
        for pair in sample_meta.pair.values])
    sums.to_csv(fn.with_suffix('.csv'))

    return sums


if __name__ == '__main__':
    import sys
    from functools import reduce
    yaml = ruamel.yaml.YAML()
    data_path = pathlib.Path(sys.argv[1]).absolute().resolve()
    print(f'Analyzing {data_path}')
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    data_df = pd.DataFrame({'path': paths})

    values = pd.DataFrame()
    iterator = tqdm(data_df.iterrows(), total=len(data_df))
    for i, row in iterator:
        iterator.set_description(row.path.stem)
        with multiprocessing.Pool() as p:
            samples = p.map(compute, row.path.glob('rank_*.npz'))

        X, W_0, W, params = load(row.path / 'rank_0.npz')
        with open(row.path / 'params.yaml', 'w') as f:
            yaml.dump(params, f)
        n_neurons = params['n_neurons']

        data_df.loc[i, params.keys()] = params.values()
        data_df.loc[i, 'sigma'] = params['glorot_normal']['sigma']

        s_W = svd(W_0, compute_uv=False)
        data_df.loc[i, 'W_condition'] = s_W.max() / s_W.min()
        data_df.loc[i, 'W_smin'] = s_W.min()
        data_df.loc[i, 'W_smax'] = s_W.max()

        x = decompress_events(X, len(W), params['n_time_step'])
        cov_x = np.cov(x[:len(W_0)])
        s_cov = svd(cov_x, compute_uv=False)
        data_df.loc[i, 'cov_condition'] = s_cov.max() / s_cov.min()
        data_df.loc[i, 'cov_smin'] = s_cov.min()
        data_df.loc[i, 'cov_smax'] = s_cov.max()


        sample = reduce_sum(samples)
        sample = pd.DataFrame([
            compute_connectivity_from_sum(row)
            for i, row in sample.iterrows()])
        sample.to_csv(row.path / 'sample.csv')
        data_df.loc[i, 'error_beta_did'] = min_error(sample, 'beta_did').fun
        data_df.loc[i, 'error_beta_iv_did'] = min_error(sample, 'beta_iv_did').fun
        data_df.loc[i, 'error_beta'] = min_error(sample, 'beta').fun
        data_df.loc[i, 'error_beta_iv'] = min_error(sample, 'beta_iv').fun

    data_df.loc[:,'error_diff'] = data_df.error_beta - data_df.error_beta_iv
    data_df.loc[:,'error_diff_did'] = data_df.error_beta_did - data_df.error_beta_iv_did
    data_df.to_csv(data_path / 'summary.csv')

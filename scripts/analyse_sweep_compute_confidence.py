import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import pandas as pd
import pathlib
from collections import defaultdict
import ruamel.yaml


from causal_optoconnectics.tools import (
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
    data = {k: data[k][()] for k in data.keys() if k in ['params', 'W', 'W_0']}
    return data

if __name__ == '__main__':
    n_iter = 100
    import sys
    rng = default_rng()
    data_path = pathlib.Path(sys.argv[1]).absolute().resolve()
    print(f'Analyzing {data_path}')
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    df = {path: defaultdict(list) for path in paths}

    values = pd.DataFrame()
    pbar = tqdm(total=len(paths)*n_iter)
    for path in paths:
        data = load(path)
        W_0 = data['W_0']
        W = data['W']
        stim_index = len(W_0)
        params = data['params'].update(rparams)
        trials = np.load(path / 'trials.npz', allow_pickle=True)['data'][()]
        for ii in range(n_iter):
            idxs = rng.choice(len(trials[0]), size=1.5e6, replace=False)
            sub_trials = {k: v[idxs] for k, v in trials.items()}
            results_meta = process_metadata(W=W, stim_index=stim_index, params=params)
            sample_meta = results_meta.query('source_stim and not target_stim and weight >= 0')
            sample = pd.DataFrame([process(pair=pair, W=W, stim_index=stim_index, trials=sub_trials, params=params) for pair in sample_meta.pair.values])
            sample.to_csv(path / f'sub_sample_{ii}.csv')
            df[path]['error_beta_did'].append(min_error(sample, 'beta_did').fun)
            df[path]['error_beta_iv_did'].append(min_error(sample, 'beta_iv_did').fun)
            df[path]['error_beta'].append(min_error(sample, 'beta').fun)
            df[path]['error_beta_iv'].append(min_error(sample, 'beta_iv').fun)
            pbar.update(1)

    pbar.close()
    np.savez(data_path / 'bootstrap.npz', data=df)

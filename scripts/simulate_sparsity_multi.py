import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support, current_process, RLock
from functools import partial
import pathlib

from causal_optoconnectics.tools import conditional_probability, joint_probability, roll_pad
from causal_optoconnectics.generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    simulate,
    construct_additional_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    generate_oscillatory_drive,
    _multiprocess_simulate,
    dales_law_transform,
    sparsify
)

def construct(params, rng=None):
    rng = default_rng() if rng is None else rng
    # set stim
    binned_stim_times = generate_poisson_stim_times(
        params['stim_period'],
        params['stim_isi_min'],
        params['stim_isi_max'],
        params['n_time_step'],
        rng=rng
    )

    binned_drive = generate_regular_stim_times(
        params['drive_period'],
        params['n_time_step']
    )
    stimulus = np.concatenate((binned_stim_times, binned_drive), 0)
    W_0 = construct_connectivity_matrix(params)
    W_0 = sparsify(W_0, params['sparsity'], rng)
    W_0 = dales_law_transform(W_0)
    W, W_0, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_additional_filters(
        W, excit_idx[:params['n_stim']], params['stim_scale'],
        params['stim_strength'])
    W = construct_additional_filters(
        W, range(len(W_0)), params['drive_scale'], params['drive_strength'])

    return W, W_0, stimulus, excit_idx, inhib_idx

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/')
    data_path.mkdir(parents=True, exist_ok=True)

    params = {
        'const': 5,
        'n_neurons': 50,
        'n_stim': 5,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'stim_scale': 2,
        'stim_strength': 5,
        'stim_period': 50,
        'stim_isi_min': 10,
        'stim_isi_max': 200,
        'drive_scale': 10,
        'drive_strength': -5,
        'drive_period': 100,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(1e6),
        'sparsity': 0.8,
        'seed': 12345,
    }
    ss = SeedSequence(params['seed'])
    num_cores = multiprocessing.cpu_count()
    child_seeds = ss.spawn(num_cores)
    rng = default_rng(params['seed'])

    fname =  f'sparsity_{sparsity:.1f}'.replace('.','')
    (data_path / fname).mkdir(exist_ok=True)

    W, W_0, stimulus, excit_idx, inhib_idx = construct(params, rng=rng)
    pool = Pool(
        initializer=tqdm.set_lock,
        initargs=(RLock(),),
        processes=num_cores
    )
    with pool as p:
        res = p.map(
            partial(
                _multiprocess_simulate,
                W=W,
                W_0=W_0,
                inputs=stimulus,
                params=params,
                pbar=True
            ),
            child_seeds)

    np.savez(
        data_path / fname,
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )

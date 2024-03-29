import numpy as np
import torch
from numpy.random import default_rng
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
    simulate_torch,
    construct_input_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    dales_law_transform,
    simulate_torch
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

    #binned_drive = generate_poisson_stim_times(
    #    params['drive_period'],
    #    params['drive_isi_min'],
    #    params['drive_isi_max'],
    #    params['n_time_step'],
    #    rng=rng
    #)
    #stimulus = np.concatenate((binned_stim_times, binned_drive), 0)
    stimulus = binned_stim_times
    W_0 = construct_connectivity_matrix(params)
    #W_0 = (np.abs(W_0.T) * np.sign(W_0.mean(1))).T
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_input_filters(
        W, excit_idx[:params['n_stim']], params['stim_scale'],
        params['stim_strength'])
    #W = construct_input_filters(
    #    W, range(len(W_0)), params['drive_scale'], params['drive_strength'])

    return W, W_0, stimulus, excit_idx, inhib_idx

def _multiprocess_simulate(i, **kwargs):
    rng = torch.Generator(device=kwargs['device'])
    rng.manual_seed(i)
    kwargs['rng'] = rng # seed each process
    # nice progressbar for each process
    from multiprocessing import current_process
    if kwargs['pbar'] is not None:
        current = current_process()
        pos = current._identity[0] - 1

        kwargs['pbar'] = partial(tqdm, position=pos)
    return simulate_torch(**kwargs)

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/')
    data_path.mkdir(parents=True, exist_ok=True)

    params = {
        'const': 5,
        'n_neurons': 10,
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
        #'drive_scale': 10,
        #'drive_strength': -6,
        #'drive_period': 100,
        #'drive_isi_min': 20,
        #'drive_isi_max': 200,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(5e6),
        'seed': 12345,
    }
    num_cores = 10
    rng = default_rng(params['seed'])

    fname =  f'n10_ss5_s5'

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
                pbar=True,
                device='cuda'
            ),
            range(params['seed'],params['seed']+num_cores))

    np.savez(
        data_path / fname,
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )

import numpy as np
from numpy.random import default_rng, SeedSequence
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
    construct_input_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    generate_oscillatory_drive,
    _multiprocess_simulate
)

if __name__ == '__main__':
    import os
    data_path = pathlib.Path('nostim-confounding-3-neurons/sweep_1')
    data_path.mkdir(parents=True, exist_ok=True)
    params = {
        'const': 5.,
        'n_neurons': 3,
        'n_stim': 5,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'drive_scale_ex': 10,
        'drive_strength_ex': 2,
        'drive_period_ex': 100,
        'drive_isi_min_ex': 30,
        'drive_isi_max_ex': 400,
        'drive_scale_in': 10,
        'drive_strength_in': -5,
        'drive_period_in': 100,
        'drive_isi_min_in': 30,
        'drive_isi_max_in': 400,
        'alpha': 0.2,
        'n_time_step': int(1e6),
        'seed': 12345
    }
    ss = SeedSequence(params['seed'])
    num_cores = multiprocessing.cpu_count()
    child_seeds = ss.spawn(num_cores)
    rng = default_rng(params['seed'])

    W_0 = np.array([
        [0, 0, 0],
        [0, 0, 2.],
        [0, 0, 0]
    ])

    binned_drive_ex = generate_poisson_stim_times(
        params['drive_period_ex'],
        params['drive_isi_min_ex'],
        params['drive_isi_max_ex'],
        params['n_time_step'],
        rng=rng
    )
    binned_drive_in = generate_poisson_stim_times(
        params['drive_period_in'],
        params['drive_isi_min_in'],
        params['drive_isi_max_in'],
        params['n_time_step'],
        rng=rng
    )
    stimulus = np.concatenate((binned_drive_ex, binned_drive_in), 0)


    for conn_strength in np.arange(0,8,1):
        W_0[1, 2] = conn_strength
        W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
        W = construct_input_filters(W, [0, 1], params['stim_scale'], params['stim_strength'])
        W = construct_input_filters(W, [0, 1, 2], params['drive_scale_ex'], params['drive_strength_ex'])
        W = construct_input_filters(W, [0, 1, 2], params['drive_scale_in'], params['drive_strength_in'])

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

        ############## SAVE ###########################
        fname = f'conn_{conn_strength}'.replace('.', ' ')

        np.savez(
            data_path / fname,
            data=res,
            W=W,
            W_0=W_0,
            params=params,
            excitatory_neuron_idx=excit_idx,
            inhibitory_neuron_idx=inhib_idx
        )
        # clear the window
        os.system('cls||clear')

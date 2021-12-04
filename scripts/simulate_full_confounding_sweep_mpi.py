import numpy as np
from numpy.random import default_rng
import pathlib
from mpi4py import MPI

from causal_optoconnectics.tools import conditional_probability, joint_probability, roll_pad
from causal_optoconnectics.generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    simulate,
    construct_input_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    generate_oscillatory_drive,
    dales_law_transform,
)

def construct(params, rng):
    # set stim
    binned_stim_times = generate_poisson_stim_times(
        params['stim_period'],
        params['stim_isi_min'],
        params['stim_isi_max'],
        params['n_time_step'],
        rng=rng
    )

    binned_drive_ex = generate_regular_stim_times(
        params['drive_period_ex'],
        params['n_time_step']
    )
    binned_drive_in = generate_regular_stim_times(
        params['drive_period_in'],
        params['n_time_step']
    )
    stimulus = np.concatenate((binned_stim_times, binned_drive_ex, binned_drive_in), 0)
    W_0 = construct_connectivity_matrix(params)
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_input_filters(
        W, excit_idx[:params['n_stim']], params['stim_scale'],
        params['stim_strength'])
    W = construct_input_filters(
        W, range(len(W_0)), params['drive_scale_ex'], params['drive_strength_ex'])
    W = construct_input_filters(
        W, range(len(W_0)), params['drive_scale_in'], params['drive_strength_in'])

    return W, W_0, stimulus, excit_idx, inhib_idx

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/sweep_8')
    data_path.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = {
        'const': 5,
        'n_neurons': None,
        'n_stim': 5,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'stim_scale': 2,
        'stim_strength': None,
        'stim_period': 50,
        'stim_isi_min': 10,
        'stim_isi_max': 200,
        'drive_scale_ex': 10,
        'drive_strength_ex': 2,
        'drive_period_ex': 100,
        'drive_scale_in': 10,
        'drive_strength_in': -5,
        'drive_period_in': 100,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': None
        },
        'n_time_step': int(1e6),
        'seed': 12345 + rank
    }
    n_neuronss = [10, 20, 30, 40, 50, 60]
    stim_strengths = [1e-6, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    sigmas = [0.5, 1, 2, 3, 4, 5, 6, 7]

    rng = default_rng(params['seed'])

    connectivity = {}

    for n_neurons in n_neuronss:
        for stim_strength in stim_strengths:
            for sigma in sigmas:
                params['glorot_normal']['sigma'] = sigma
                params['stim_strength'] = stim_strength
                params['n_neurons'] = n_neurons
                path =  f'n{n_neurons}_ss{stim_strength:.2f}_s{sigma}'.replace('.','')
                if (data_path / path).exists():
                    continue
                (data_path / path).mkdir(exist_ok=True)
                fname = data_path / path/ f'rank_{rank}.npz'

                if rank == 0:
                    connectivity[path] = construct(params, rng=rng)

                connectivity = comm.bcast(connectivity, root=0)
                W, W_0, stimulus, excit_idx, inhib_idx = connectivity[path]
                res = simulate(W=W, W_0=W_0, inputs=stimulus, params=params, rng=rng)

                np.savez(
                    fname,
                    data=res,
                    W=W,
                    W_0=W_0,
                    params=params,
                    excitatory_neuron_idx=excit_idx,
                    inhibitory_neuron_idx=inhib_idx
                )

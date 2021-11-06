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
    stimulus = generate_poisson_stim_times(
        params['stim_period'],
        params['stim_isi_min'],
        params['stim_isi_max'],
        params['n_time_step'],
        rng=rng
    )
    W_0 = construct_connectivity_matrix(params)
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_input_filters(
        W, excit_idx[:params['n_stim']], params['stim_scale'],
        params['stim_strength'])

    return W, W_0, stimulus, excit_idx, inhib_idx

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/sweep_1')
    data_path.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(1e6),
        'seed': 12345 + rank
    }
    n_neurons = params['n_neurons']
    stim_strength = params['stim_strength']
    sigma = params['sigma']
    rng = default_rng(params['seed'])

    data_path.mkdir(parents=True, exist_ok=True)

    connectivity = {}

    path =  f'n{n_neurons}_ss{stim_strength}_s{sigma}'.replace('.','')
    (data_path / path).mkdir(exist_ok=True)

    if rank == 0:
        connectivity[path] = construct(params)

    connectivity = comm.bcast(connectivity, root=0)
    W, W_0, stimulus, excit_idx, inhib_idx = connectivity[path]
    res = simulate(W=W, W_0=W_0, inputs=stimulus, params=params, rng=rng)

    np.savez(
        data_path / path / f'rank_{rank}',
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )

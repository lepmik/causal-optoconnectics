import numpy as np
import pathlib
from mpi4py import MPI

from causal_optoconnectics.tools import conditional_probability, joint_probability, roll_pad
from causal_optoconnectics.generator import (
    construct_connectivity_filters,
    construct_connectivity_matrix,
    simulate,
    construct_additional_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    generate_oscillatory_drive,
    dales_law_transform,
)

def construct(params, sparsity=0):
    # set stim
    binned_stim_times = generate_poisson_stim_times(
        params['stim_period'],
        params['stim_isi_min'],
        params['stim_isi_max'],
        params['n_time_step']
    )

    binned_drive = generate_regular_stim_times(
        params['drive_period'],
        params['n_time_step']
    )
    stimulus = np.concatenate((binned_stim_times, binned_drive), 0)
    W_0 = construct_connectivity_matrix(params)
    indices = np.unravel_index(
        np.random.choice(
            np.arange(np.prod(W_0.shape)),
            size=int(sparsity*np.prod(W_0.shape)),
            replace=False
        ),
        W_0.shape
    )
    W_0[indices] = 0
    W_0 = dales_law_transform(W_0)
    W, W_0, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_additional_filters(
        W, excit_idx[:params['n_stim']], params['stim_scale'],
        params['stim_strength'])
    W = construct_additional_filters(
        W, range(len(W_0)), params['drive_scale'], params['drive_strength'])

    return W, W_0, stimulus

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/sweep_3')
    data_path.mkdir(parents=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    np.random.seed()
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
        'n_time_step': int(1e6)
    }
    sparsities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if rank == 0:
        connectivity = {}
    else:
        connectivity = None
    for sparsity in sparsities:
        path =  f'sparsity_{sparsity:.1f}'.replace('.','')
        (data_path / path).mkdir(exist_ok=True)
        if rank == 0:
            W, W_0, stimulus = construct(params, sparsity=sparsity)
            connectivity[path] = (W, W_0, stimulus)
        connectivity = comm.bcast(connectivity, root=0)
        W, W_0, stimulus = connectivity[path]
        res = simulate(W=W, W_0=W_0, inputs=stimulus, params=params)
        np.savez(
            data_path / path/ f'rank_{rank}',
            data=res,
            W=W,
            W_0=W_0,
            params=params,
            excitatory_neuron_idx=excit_idx,
            inhibitory_neuron_idx=inhib_idx
        )

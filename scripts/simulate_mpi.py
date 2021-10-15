import numpy as np
from functools import partial
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
)

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = {
        'const': 5.,
        'n_neurons': 10,
        'n_stim': 5,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'stim_scale': 2,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'stim_strength': 4,
        'alpha': 0.2,
        'glorot_normal': {
            'mu': 0,
            'sigma': 6
        },
        'n_time_step': int(1e6)
    }
    n_neurons = params['n_neurons']

    data_path.mkdir(parents=True, exist_ok=True)
    fname = 'rank_{}_{}_{}_{}'.format(
        rank,
        params['n_neurons'],
        params['stim_strength'],
        params['glorot_normal']['sigma']
    )

    assert not (data_path / fname).exists()
    if rank == 0:
        W, W_0, excit_idx, inhib_idx = construct_connectivity_matrix(params)
        connectivity = (W, W_0, excit_idx, inhib_idx)
    else:
        connectivity = None
    connectivity = comm.bcast(connectivity, root=0)
    W, W_0, excit_idx, inhib_idx = connectivity

    res = simulate(W=W, params=params)

    np.savez(
        data_path / fname,
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )

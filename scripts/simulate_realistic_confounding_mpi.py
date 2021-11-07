import numpy as np
from numpy.random import default_rng
import pathlib
from mpi4py import MPI

from causal_optoconnectics.tools import conditional_probability, joint_probability, roll_pad
from causal_optoconnectics.generator import (
    construct_connectivity_filters,
    simulate,
    construct_input_filters,
    generate_poisson_stim_times,
    generate_regular_stim_times,
    generate_oscillatory_drive,
    dales_law_transform,
    sparsify,
    clipped_lognormal
)

def construct_connectivity_matrix(params):
    W_ex = clipped_lognormal(
        mu=params['lognormal']['mu_ex'],
        sigma=params['lognormal']['sigma_ex'],
        size=(params['n_ex_neurons'], params['n_neurons']),
        low=params['lognormal']['low_ex'],
        high=params['lognormal']['high_ex'],
    )
    W_in = clipped_lognormal(
        mu=params['lognormal']['mu_in'],
        sigma=params['lognormal']['sigma_in'],
        size=(params['n_ex_neurons'], params['n_neurons']),
        low=params['lognormal']['low_in'],
        high=params['lognormal']['high_in'],
    )
    W_ex = sparsify(W_ex, params['sparsity_ex'], rng)
    W_in = sparsify(W_in, params['sparsity_in'], rng)
    W_0 = np.concatenate([W_ex, -W_in], 0)
    np.fill_diagonal(W_0, 0)
    return W_0


def compute_stim_amps(params, nodes, rng):
    def intensity(z):
        rho = params['r'] * np.sqrt((params['n'] / params['NA'])**2 - 1)
        return rho**2 / ((params['S'] * z + 1) * (z + rho)**2)

    def affected_neurons(z):
        theta = np.arcsin(params['NA'] / params['n'])
        lcorr = params['r'] / np.tan(theta)
        rad = (z + lcorr) * np.tan(theta)
        A = np.pi * rad**2
        dz = z[1] - z[0]
        dV = A * dz
        density = params['stim_n_ex'] / sum(dV)
        params['density'] = density
        N = dV * density
        return N

    def hill(I):
        In = I**params['n_hill']
        return params['Imax'] * In / (params['K']**params['n_hill'] + In) # peak amplitude of the current response

    # Set dc stimulation
    z = np.linspace(0, params['depth'], params['n_pos'])
    n_slice = affected_neurons(z).astype(int)
    I = intensity(z)
    A = hill(params['I0'] * I)
    A = A / A.max()
    idx = 0
    stim_amps = {}
    for i, n_stim in enumerate(n_slice):
        amp = A[i] * params['stim_amp_ex']
        stim_amps.update({n: amp for n in nodes[idx:idx + n_stim]})
        idx += n_stim
    return stim_amps


def construct(params, rng):
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
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    stim_amps = compute_stim_amps(params)
    W = construct_input_filters(
        W, stim_amps.keys(), params['stim_scale'], stim_amps)

    W = construct_input_filters(
        W, range(len(W_0)), params['drive_scale'], params['drive_strength'])

    return W, W_0, stimulus, excit_idx, inhib_idx

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/simulation_1/realistic')
    data_path.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = {
        'const': 5,
        'n_neurons': 10000,
        'n_ex_neurons': 8000,
        'n_in_neurons': 2000,
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
        'stim_n_ex': 1000,
        'drive_scale': 10,
        'drive_strength': -10,
        'drive_period': 100,
        'alpha': 0.2,
        'sparsity_ex': 0.9,
        'sparsity_in': 0.6, # balanced network, n_synapses = 800
        'lognormal': {
            'mu_ex': 2,
            'sigma_ex': 4,
            'low_ex': 0,
            'high_ex': 5,
            'mu_in': 2,
            'sigma_in': 4,
            'low_in': 0,
            'high_in': 5
        },
        'n_time_step': int(1e6),
        'seed': 12345 + rank,
        # Optogenetics
        'I0': 10, # light intensity leaving fibre mW/mm2
        'r': 100e-3, # 100 um radius of fiber
        'n': 1.36, # refraction index of gray matter
        'NA': 0.37, # Numerical Aperture of fiber
        'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
        'n_pos': 100,
        'depth': .7,
        'Imax': 642, # max current pA
        'K': 0.84, # half-maximal light sensitivity of the ChR2 mW/mm2
        'n_hill': 0.76, # Hill coefficient
    }

    rng = default_rng(params['seed'])

    connectivity = {}

    fname = data_path / f'rank_{rank}.npz'

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

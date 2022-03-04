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
    dales_law_transform,
    sparsify
)


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
        density = params['n_stim'] / sum(dV)
        params['density'] = float(density)
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
        amp = A[i] * params['stim_strength']
        stim_amps.update({n: amp for n in nodes[idx:idx + n_stim]})
        idx += n_stim
    return stim_amps


def construct(params, rng):
    # set stim
    stimulus = generate_poisson_stim_times(
        params['stim_period'],
        params['stim_isi_min'],
        params['stim_isi_max'],
        params['n_time_step'],
        rng=rng
    )

    W_0 = construct_connectivity_matrix(params)
    W_0 = sparsify(W_0, params['sparsity'], rng=rng)
    W_0 = dales_law_transform(W_0)
    W, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)

    stim_amps = compute_stim_amps(params, range(params['n_neurons']), rng)
    W = construct_input_filters(
        W, stim_amps.keys(), params['stim_scale'], stim_amps)

    return W, W_0, stimulus, excit_idx, inhib_idx, stim_amps

if __name__ == '__main__':
    data_path = pathlib.Path('datasets/sweep_7_ss6_np15')
    data_path.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = {
        'const': 5,
        'n_neurons': None,
        'dt': 1e-3,
        'ref_scale': 10,
        'abs_ref_scale': 3,
        'spike_scale': 5,
        'abs_ref_strength': -100,
        'rel_ref_strength': -30,
        'stim_scale': 2,
        'stim_strength': 6,
        'stim_period': 50,
        'stim_isi_min': 10,
        'stim_isi_max': 200,
        'n_stim': None,
        'alpha': 0.2,
        'sparsity': 0,
        'glorot_normal': {
            'mu': 0,
            'sigma': 5
        },
        'n_time_step': int(1e6),
        'seed': 12345 + rank,
        # Optogenetics
        'I0': 10, # light intensity leaving fibre mW/mm2
        'r': 100e-3, # 100 um radius of fiber
        'n': 1.36, # refraction index of gray matter
        'NA': 0.37, # Numerical Aperture of fiber
        'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
        'n_pos': 15,
        'depth': .7,
        'Imax': 642, # max current pA
        'K': 0.84, # half-maximal light sensitivity of the ChR2 mW/mm2
        'n_hill': 0.76, # Hill coefficient
    }

    rng = default_rng(params['seed'])


    connectivity = {}

    for n_neurons in [50, 75, 100, 150, 200, 250]:
        path =  data_path / f'realistic_n{n_neurons}'
        params.update({
            'n_neurons': n_neurons,
            'n_stim': n_neurons,
#             'n_pos': int(n_neurons / 6.),
            
        })
        if path.exists():
            continue
        if rank == 0:
            connectivity[path] = construct(params, rng=rng)
        comm.Barrier()
        path.mkdir(exist_ok=True)
        fname = path / f'rank_{rank}.npz'

        connectivity = comm.bcast(connectivity, root=0)
        W, W_0, stimulus, excit_idx, inhib_idx, stim_amps = connectivity[path]
        res = simulate(W=W, W_0=W_0, inputs=stimulus, params=params, rng=rng)

        np.savez(
            fname,
            data=res,
            stim_amps=stim_amps,
            W=W,
            W_0=W_0,
            params=params,
            excitatory_neuron_idx=excit_idx,
            inhibitory_neuron_idx=inhib_idx
        )

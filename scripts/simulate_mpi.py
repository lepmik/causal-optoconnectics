import numpy as np
from numpy.random import default_rng
import pathlib
from mpi4py import MPI
from causal_optoconnectics.generator import simulate


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 3
    data_path = pathlib.Path(sys.argv[1]).absolute().resolve()
    set_seed = int(sys.argv[2])
    print('SYS.ARGV', sys.argv[1:])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    settings = None
    paths = [path for path in data_path.iterdir() if path.is_dir()]
    for path in paths:
        if rank == 0:
            data = np.load(path / 'rank_0.npz', allow_pickle=True)
            params, W, W_0, stimulus, excit_idx, inhib_idx = map(
                data.get,
                ['params', 'W', 'W_0', 'stimulus', 'excitatory_neuron_idx', 'inhibitory_neuron_idx']
            )
            base_rank = max([int(fn.stem.split('_')[-1]) for fn in path.glob('rank_*.npz')])
            seeds = [np.load(fn, allow_pickle=True)['params'][()]['seed'] for fn in path.glob('rank_*.npz')]
            settings = (params[()], W, W_0, stimulus, excit_idx, inhib_idx, base_rank, seeds)
        params, W, W_0, stimulus, excit_idx, inhib_idx, base_rank, seeds = comm.bcast(settings, root=0)
        seed = set_seed + rank
        assert seed not in seeds
        params['seed'] = seed
        rng = default_rng(seed)

        W, W_0, stimulus, excit_idx, inhib_idx = connectivity[path]
        res = simulate(W=W, W_0=W_0, inputs=stimulus, params=params, rng=rng)

        np.savez(
            data_path / path / f'rank_{rank + base_rank}',
            data=res,
            W=W,
            W_0=W_0,
            params=params,
            excitatory_neuron_idx=excit_idx,
            inhibitory_neuron_idx=inhib_idx
        )

import scipy.stats as st
import numpy as np
from tqdm import tqdm
from functools import partial
import pathlib
from .tools import roll_pad


def clipped_lognormal(mu, sigma, size, low, high):
    sample = np.random.lognormal(mu, sigma, size)
    while ((sample < low) | (sample > high)).any():
        mask = list(np.where((sample < low) | (sample > high)))
        subsample = np.random.lognormal(mu, sigma, size)
        submask = list(np.where((subsample > low) & (subsample < high)))
        n = min(len(mask[0]), len(submask[0]))
        for i in range(len(mask)):
            mask[i] = mask[i][:n]
            submask[i] = submask[i][:n]
        sample[tuple(mask)] = subsample[tuple(submask)]

    return sample


def clipped_poisson(mu, size, low, high, max_iter=100000):
    truncated = st.poisson.rvs(mu, size=size)
    itr = 0
    while ((truncated < low) | (truncated > high)).any():
        mask, = np.where((truncated < low) | (truncated > high))
        temp = st.poisson.rvs(mu, size=size)
        temp_mask, = np.where((temp >= low) & (temp <= high))
        mask = mask[:len(temp_mask)]
        truncated[mask] = temp[:len(mask)]
        itr += 1
        if itr > max_iter:
            print('Did not reach the desired limits in "max_iter" iterations')
            return None
    return truncated


def simulate_simple_conditional_response(stim_times, make_post=False, response='gaussian', **p):
    n_stim = len(stim_times)
    idxs = np.random.permutation(np.arange(n_stim).astype(int))
    n_stim_spikes = int(n_stim * p['stim_hit_chance'])
    idxs_stim_spikes = idxs[:n_stim_spikes]
    noise_x = np.random.uniform(0, p['stop_time'], p['pre_rate'] * p['stop_time'])
    if response == 'gaussian':
        response_x = st.norm.rvs(
            loc=stim_times[idxs_stim_spikes] + p['stim_latency'],
            scale=p['stim_latency_std'])
    elif response == 'fixed':
        response_x = stim_times[idxs_stim_spikes] + p['stim_latency']
    spikes = np.sort(np.concatenate([response_x, noise_x]))
    pre_spikes = prune(spikes, p['refractory'])
    n_pre_spikes = len(pre_spikes)
    if make_post:
        n_post_spikes = int(n_pre_spikes * p['pre_hit_chance'])
        idxs_post_spikes = np.random.permutation(np.arange(n_pre_spikes).astype(int))[:n_post_spikes]
        noise_y = np.random.uniform(
            0, p['stop_time'], int(p['post_rate'] * p['stop_time']))
        if response == 'gaussian':
            response_y = st.norm.rvs(
                loc=pre_spikes[idxs_post_spikes] + p['latency'],
                scale=p['latency_std'])
        elif response == 'fixed':
            response_y = pre_spikes[idxs_post_spikes] + p['latency']
        post_spikes = np.sort(np.concatenate([response_y, noise_y]))
        post_spikes = prune(post_spikes, p['refractory'])

        return pre_spikes, post_spikes
    else:
        return pre_spikes


def construct_connectivity_matrix(params):
    if 'uniform' in params:
        W_0 = np.random.uniform(
            low=params['uniform']['low'],
            high=params['uniform']['high'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'normal' in params:
        W_0 = np.random.normal(
            loc=params['normal']['mu'],
            scale=params['normal']['sigma'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'glorot_normal' in params:
        W_0 = np.random.normal(
            loc=params['glorot_normal']['mu'],
            scale=params['glorot_normal']['sigma'] / np.sqrt(params['n_neurons']),
            size=(params['n_neurons'], params['n_neurons'])
        )
    else:
        raise ValueError()

    return W_0


def dales_law_transform(W_0):
    # Dale's law
    W_0 = np.concatenate((W_0*(W_0>0), W_0*(W_0<0)), 0)
    W_0 = np.concatenate((W_0, W_0), 1)
    return W_0


def construct_connectivity_filters(W_0, params):
    np.fill_diagonal(W_0, np.nan)
    # construct construct connectivity matrix
    W = np.zeros((W_0.shape[0], W_0.shape[1], params['ref_scale']))
    for i in range(W_0.shape[0]):
        for j in range(W_0.shape[1]):
            if i==j:
                W[i, j, :params['abs_ref_scale']] = params['abs_ref_strength']
                abs_ref = np.arange(params['abs_ref_scale'], params['ref_scale'])
                W[i, j, params['abs_ref_scale']:params['ref_scale']] = \
                    params['rel_ref_strength'] * np.exp(-0.5*(abs_ref+4))
            else:
                W[i, j, np.arange(params['spike_scale'])] = \
                    W_0[i,j] * \
                    np.exp(-params['alpha']*np.arange(params['spike_scale']))

    excitatory_neuron_idx, = np.where(np.any(W_0 > 0, 1))
    inhibitory_neuron_idx, = np.where(np.any(W_0 < 0, 1))

    return W, W_0, excitatory_neuron_idx, inhibitory_neuron_idx


def generate_regular_stim_times(period, size):
    binned_stim_times = np.zeros(size)
    binned_stim_times[np.arange(period, size, period)] = 1
    binned_stim_times = np.expand_dims(binned_stim_times, 0)
    return binned_stim_times


def generate_poisson_stim_times(period, low, high, size):
    isi = []
    while sum(isi) < size:
        isi += clipped_poisson(period, 100, low, high).tolist()
    cum = np.cumsum(isi)
    cum = cum[cum < size].astype(int)
    binned_stim_times = np.zeros(size)
    binned_stim_times[cum] = 1
    return np.expand_dims(binned_stim_times, 0)


def construct_additional_filters(W, indices, scale, strength):
    W = np.concatenate((W, np.zeros((1, W.shape[1], W.shape[2]))), 0)
    W = np.concatenate((W, np.zeros((W.shape[0], 1, W.shape[2]))), 1)#TODO do we really need this one??
    for j in indices:
        W[-1, j, np.arange(scale)] = strength
    return W


def generate_oscillatory_drive(params):
    t = np.arange(params['n_time_step']) * params['dt']
    binned_stim_times = (np.sin(2*np.pi*t*params['drive_freq']) > 0).astype(int)
    binned_stim_times = np.expand_dims(binned_stim_times, 0)
    return binned_stim_times


def simulate(W, W_0, inputs, params, pbar=None):
    pbar = pbar if pbar is not None else lambda x:x
    x = np.zeros((len(W), params['ref_scale']))
    rand_init = np.random.randint(0, 2, params['n_neurons'])
    # if W_0 has dales law transform n_neurons = len(W_0) / 2 and the first
    # half of neurons are excitatory and the second half is their inhibitory
    # copies and thus have to be identically initialized
    if len(W_0) == params['n_neurons'] * 2:
        rand_init = np.concatenate((rand_init, rand_init))

    x[:len(W_0), -1] = rand_init

    x[len(W_0):] = inputs[:, :x.shape[1]]
    spikes = []

    for t in pbar(range(params['n_time_step'] - 1)):

        if t >= params['ref_scale']:
            x[len(W_0):] = inputs[:, t-params['ref_scale']+1: t+1]

        # if any spikes store spike indices and time
        if x[:,-1].any():
            spikes.extend([(idx, t) for idx in np.where(x[:,-1])[0]])

        activation = np.dot(W.T, x)

        activation = activation[
            np.arange(params['ref_scale']),
            :,
            np.arange(params['ref_scale'])[::-1]
        ].sum(0)

        activation = activation - params['const']

        #Stimulus has no activation
        activation = activation[:len(W_0)]

        x = roll_pad(x, -1)

        x[:len(W_0), -1] = np.random.binomial(
            1, np.exp(activation) / (np.exp(activation) + 1), size=len(W_0)
        ) # binomial GLM with logit link function (binomial regression)
    return np.array(spikes)


def _multiprocess_simulate(i, **kwargs):
    np.random.seed()
    if kwargs['pbar'] is not None:
        current = current_process()
        pos = current._identity[0] - 1

        kwargs['pbar'] = partial(tqdm, position=pos)
    return simulate(**kwargs)


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool, freeze_support, current_process, RLock
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
        # 'normal': {
        #     'mu': 0,
        #     'sigma': 5
        # },
        # 'uniform': {
        #     'low': 0,
        #     'high': 1
        # },
        'n_time_step': int(1e6)
    }
    n_neurons = params['n_neurons']

    data_path = pathlib.Path('datasets/')
    data_path.mkdir(parents=True, exist_ok=True)
    fname = 'poisson_multi_simulation_{}_{}_{}'.format(
        params['n_neurons'],
        params['stim_strength'],
        params['glorot_normal']['sigma']
    )

    assert not (data_path / fname).exists()

    num_cores = multiprocessing.cpu_count()

    W_0 = construct_connectivity_matrix(params)
    W_0 = dales_law_transform(W_0)
    W, W_0, excit_idx, inhib_idx = construct_connectivity_filters(W_0, params)
    W = construct_additional_filters(
        W, excit_idx[:params['n_stim']],
        params['stim_scale'], params['stim_strength']
    )
    stimulus = generate_regular_stim_times(params)
    freeze_support()
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
            range(num_cores))

    np.savez(
        data_path / fname,
        data=res,
        W=W,
        W_0=W_0,
        params=params,
        excitatory_neuron_idx=excit_idx,
        inhibitory_neuron_idx=inhib_idx
    )

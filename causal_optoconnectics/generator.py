import scipy.stats as st
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from functools import partial
import pathlib
from .tools import roll_pad


def clipped_lognormal(mu, sigma, size, low, high, rng=None, max_iter=100000):
    rng = default_rng() if rng is None else rng
    sample = rng.lognormal(mu, sigma, size)
    while ((sample < low) | (sample > high)).any():
        mask = list(np.where((sample < low) | (sample > high)))
        subsample = rng.lognormal(mu, sigma, size)
        submask = list(np.where((subsample > low) & (subsample < high)))
        n = min(len(mask[0]), len(submask[0]))
        for i in range(len(mask)):
            mask[i] = mask[i][:n]
            submask[i] = submask[i][:n]
        sample[tuple(mask)] = subsample[tuple(submask)]
        if itr > max_iter:
            print('Did not reach the desired limits in "max_iter" iterations')
            return None
    return sample


def clipped_poisson(mu, size, low, high, max_iter=100000, rng=None):
    rng = default_rng() if rng is None else rng
    truncated = rng.poisson(mu, size=size)
    itr = 0
    while ((truncated < low) | (truncated > high)).any():
        mask, = np.where((truncated < low) | (truncated > high))
        temp = rng.poisson(mu, size=size)
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


def construct_connectivity_matrix(params, rng=None, self_connections=False):
    rng = default_rng() if rng is None else rng
    if 'uniform' in params:
        W_0 = rng.uniform(
            low=params['uniform']['low'],
            high=params['uniform']['high'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'normal' in params:
        W_0 = rng.normal(
            loc=params['normal']['mu'],
            scale=params['normal']['sigma'],
            size=(params['n_neurons'], params['n_neurons'])
        )
    elif 'glorot_normal' in params:
        W_0 = rng.normal(
            loc=params['glorot_normal']['mu'],
            scale=params['glorot_normal']['sigma'] / np.sqrt(params['n_neurons']),
            size=(params['n_neurons'], params['n_neurons'])
        )
    # elif 'lognormal' in params:
    #     mu, sigma, size, low, high
    #     W_0 = clipped_lognormal(
    #         mu=params['lognormal']['mu'],
    #         sigma=params['lognormal']['sigma'] / np.sqrt(params['n_neurons']),
    #         size=(params['n_neurons'], params['n_neurons']),
    #         low
    #         high
    #     )
    else:
        raise ValueError()
    if not self_connections:
        np.fill_diagonal(W_0, 0)
    return W_0


def distance_wrapped(i, j, length):
    a = (length / 2)
    d = abs(i - j)
    d[d > a] = abs(d[d > a] - length)
    return d

def mexican_hat(i, j, a, sigma_1, sigma_2, n_neurons):
    d = distance_wrapped(i, j, n_neurons)
    first = np.exp(- d**2 / (2 * sigma_1**2))
    second = a * np.exp(- d**2 / (2 * sigma_2**2))
    return first - second

def construct_mexican_hat_connectivity(params):
    W = np.zeros((params['n_neurons'],params['n_neurons']))
    j = np.arange(params['n_neurons'])
    for i in range(params['n_neurons']):
        W[i,j] = mexican_hat(i, j, *map(params.get, ['mex_a', 'mex_sigma_1', 'mex_sigma_2', 'n_neurons']))
    np.fill_diagonal(W, 0)
    return W


def dales_law_transform(W_0):
    # Dale's law
    W_0 = np.concatenate((W_0*(W_0>0), W_0*(W_0<0)), 0)
    W_0 = np.concatenate((W_0, W_0), 1)
    return W_0


def sparsify(W_0, sparsity, rng)=None:
    rng = default_rng() if rng is None else rng
    indices = np.unravel_index(
        rng.choice(
            np.arange(np.prod(W_0.shape)),
            size=int(sparsity * np.prod(W_0.shape)),
            replace=False
        ),
        W_0.shape
    )
    W_0[indices] = 0
    return W_0


def construct_connectivity_filters(W_0, params):
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

    return W, excitatory_neuron_idx, inhibitory_neuron_idx


def generate_regular_stim_times(period, size):
    binned_stim_times = np.zeros(size)
    binned_stim_times[np.arange(period, size, period)] = 1
    binned_stim_times = np.expand_dims(binned_stim_times, 0)
    return binned_stim_times


def generate_poisson_stim_times(period, low, high, size, rng=None):
    isi = []
    while sum(isi) < size:
        isi += clipped_poisson(period, 100, low, high, rng=rng).tolist()
    cum = np.cumsum(isi)
    cum = cum[cum < size].astype(int)
    binned_stim_times = np.zeros(size)
    binned_stim_times[cum] = 1
    return np.expand_dims(binned_stim_times, 0)


def construct_additional_filters(W, indices, scale, strength):
    W = np.concatenate((W, np.zeros((1, W.shape[1], W.shape[2]))), 0)
    # W = np.concatenate((W, np.zeros((W.shape[0], 1, W.shape[2]))), 1)#TODO do we really need this one??
    for j in indices:
        W[-1, j, np.arange(scale)] = strength
    return W


def generate_oscillatory_drive(params):
    t = np.arange(params['n_time_step']) * params['dt']
    binned_stim_times = (np.sin(2*np.pi*t*params['drive_freq']) > 0).astype(int)
    binned_stim_times = np.expand_dims(binned_stim_times, 0)
    return binned_stim_times


def simulate(W, W_0, params, inputs=None, pbar=None, rng=None):
    rng = default_rng() if rng is None else rng
    pbar = pbar if pbar is not None else lambda x:x
    x = np.zeros((len(W), params['ref_scale']))
    rand_init = rng.integers(0, 2, params['n_neurons'])
    # if W_0 has dales law transform n_neurons = len(W_0) / 2 and the first
    # half of neurons are excitatory and the second half is their inhibitory
    # copies and thus have to be identically initialized
    if len(W_0) == params['n_neurons'] * 2:
        rand_init = np.concatenate((rand_init, rand_init))

    x[:len(W_0), -1] = rand_init

    if inputs is not None:
        x[len(W_0):] = inputs[:, :x.shape[1]]

    spikes = []
    ref_scale_range = np.arange(params['ref_scale'])

    for t in pbar(range(params['n_time_step'] - 1)):

        if t >= params['ref_scale'] and inputs is not None:
            x[len(W_0):] = inputs[:, t-params['ref_scale']+1: t+1]

        # if any spikes store spike indices and time
        if x[:,-1].any():
            spikes.extend([(idx, t) for idx in np.where(x[:,-1])[0]])

        activation = np.dot(W.T, x)

        activation = activation[ref_scale_range,:,ref_scale_range[::-1]].sum(0)

        activation = activation - params['const']

        #Stimulus has no activation
        activation = activation[:len(W_0)]

        x = roll_pad(x, -1)

        x[:len(W_0), -1] = rng.binomial(
            1, np.exp(activation) / (np.exp(activation) + 1), size=len(W_0)
        ) # binomial GLM with logit link function (binomial regression)
    return np.array(spikes)


def simulate_torch(W, W_0, params, inputs=None, pbar=None, device='cpu', rng=None):
    import torch
    rng = torch.Generator() if rng is None else rng
    pbar = pbar if pbar is not None else lambda x:x

    W = torch.as_tensor(W).to(device, dtype=torch.float32)
    if inputs is not None:
        inputs = torch.as_tensor(inputs, dtype=torch.float32).to(device)

    x = torch.zeros((len(W), params['ref_scale']), dtype=torch.float32, device=device)
    rand_init = torch.randint(0, 2, (params['n_neurons'],), generator=rng, device=device)
    # if W_0 has dales law transform n_neurons = len(W_0) / 2 and the first
    # half of neurons are excitatory and the second half is their inhibitory
    # copies and thus have to be identically initialized
    if len(W_0) == params['n_neurons'] * 2:
        rand_init = torch.cat((rand_init, rand_init))

    x[:len(W_0), -1] = rand_init

    if inputs is not None:
        x[len(W_0):] = inputs[:, :x.shape[1]]
    spikes = []


    ref_scale_range = torch.arange(params['ref_scale']).to(device)
    ref_scale_range_flip = ref_scale_range.flip(0)
    for t in pbar(range(params['n_time_step'] - 1)):

        if t >= params['ref_scale'] and inputs is not None:
            x[len(W_0):] = inputs[:, t-params['ref_scale']+1: t+1]

        # if any spikes store spike indices and time
        if x[:,-1].any():
            spikes.extend([(idx.cpu(), t) for idx in torch.where(x[:,-1])[0]])

        activation = torch.einsum('kji,kl->ijl',W,x)

        activation =  activation[ref_scale_range,:,ref_scale_range_flip].sum(0)

        activation = activation - params['const']

        #Stimulus has no activation
        activation = activation[:len(W_0)]

        x = torch.roll(x, -1, 0)

        x[:len(W_0), -1] = torch.bernoulli(
            torch.exp(activation) / (torch.exp(activation) + 1),
            generator=rng
        ) # binomial GLM with logit link function (binomial regression)
    return torch.tensor(spikes).cpu().numpy()


def _multiprocess_simulate(i, **kwargs):
    kwargs['rng'] = default_rng(i) # seed each process
    # nice progressbar for each process
    from multiprocessing import current_process
    if kwargs['pbar'] is not None:
        current = current_process()
        pos = current._identity[0] - 1

        kwargs['pbar'] = partial(tqdm, position=pos)
    return simulate(**kwargs)

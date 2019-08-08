import scipy.stats as st
import numpy as np


def prune(a, ref):
    b = np.concatenate(([False], np.diff(a) < ref))
    c = np.concatenate(([False], np.diff(b.astype(int)) > 0))
    d = a[~c]
    if any(np.diff(a) < ref):
        d = prune(d, ref)
    return d


def generate_stim_times(stim_rate, stim_isi_min, stop_time):
    stim_times = np.sort(np.random.uniform(
        0, stop_time, stim_rate * stop_time))
    return prune(stim_times, stim_isi_min)


def generate_neurons(stim_times, make_post=False, response='gaussian', **p):
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

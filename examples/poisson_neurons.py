import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from causal_optoconnectics import causal_connectivity
from causal_optoconnectics.cch import fit_latency
from generator import generate_neurons, generate_stim_times
from tqdm import tqdm


np.random.seed(1234)

response = 'gaussian'
# response = 'fixed'

stim_params = {
    'stop_time': 2000, # seconds
    'stim_rate': 30, # rate of stimulation (gets reduced by pruning for minimum inter stimulus interval)
    'stim_isi_min': 30e-3, # minimum inter stimulus interval
}
neuron_params = {
    'refractory': 4e-3, # 4 ms
    'latency': 4e-3, # post response delay
    'latency_std': 1e-3,
    'pre_hit_chance': .8, # fraction of spikes that are driven by the presynaptic neuron
    'post_rate': 5, # Hz
    'pre_rate': 5, # base rate
    'stim_hit_chance': .8, # fraction of spikes that are driven by the stimulation
    'stim_latency': 2e-3, # latency from stim to pre response
    'stim_latency_std': .5e-3,
    'stop_time': stim_params['stop_time'],
}
iv_params = {
    'x_mu': 2e-3,
    'x_sigma': 1e-3,
    'y_mu': 6e-3,
    'y_sigma': 2e-3,
    'n_bases': 20,
    'bin_size': 1e-3,
    'offset': 1e-2
}

stim_times = generate_stim_times(**stim_params)

A_spikes, C_spikes = generate_neurons(
    stim_times, make_post=True, response=response, **neuron_params)

B_spikes = generate_neurons(
    stim_times, make_post=False, response=response, **neuron_params)

beta_AC = causal_connectivity(
    x=A_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

beta_BC = causal_connectivity(
    x=B_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

print(beta_AC, beta_BC)

plt.figure()
plt.title('Upstream response')
fit_latency(stim_times, A_spikes, limit=12e-3, plot=True)

plt.figure()
plt.title('Downstream response')
fit_latency(stim_times, C_spikes, limit=12e-3, plot=True)

results_hit_chance = []
for h in tqdm(np.arange(0, 1.1, .1)):
    neuron_params['pre_hit_chance'] = h

    stim_times = generate_stim_times(**stim_params)

    A_spikes, C_spikes = generate_neurons(
        stim_times, make_post=True, response=response, **neuron_params)

    B_spikes = generate_neurons(
        stim_times, make_post=False, response=response, **neuron_params)

    beta_AC = causal_connectivity(
        x=A_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    beta_BC = causal_connectivity(
        x=B_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    res = {
        'A_rate': len(A_spikes) / neuron_params['stop_time'],
        'B_rate': len(B_spikes) / neuron_params['stop_time'],
        'C_rate': len(C_spikes) / neuron_params['stop_time'],
        'S_rate': len(stim_times) / neuron_params['stop_time'],
        'C_induced_rate': (len(C_spikes) / neuron_params['stop_time']) - neuron_params['post_rate'],
        'beta_AC': beta_AC,
        'beta_BC': beta_BC
    }
    results_hit_chance.append({**stim_params, **neuron_params, **iv_params, **res})

results_hit_chance = pd.DataFrame(results_hit_chance)

keys = [
    'beta_AC',
    'beta_BC',
]

plt.figure()
cmap = cm.get_cmap('tab10')
cnt = 0
for key in keys:
    plt.plot(results_hit_chance['pre_hit_chance'], results_hit_chance[key], label=key, color=cmap(cnt))
    cnt += 1
plt.grid(True)
AC_ground_truth = results_hit_chance.C_induced_rate / (
    results_hit_chance.A_rate + results_hit_chance.S_rate)

plt.plot(results_hit_chance['pre_hit_chance'], AC_ground_truth, '--k')
plt.title('Three Poisson neurons with {} response'.format(response))
plt.xlabel('Connection strength')
plt.legend()


results_stop_time = []
for h in tqdm(np.arange(100, 6100, 10)):
    stim_params['stop_time'] = h
    neuron_params['stop_time'] = stim_params['stop_time']

    stim_times = generate_stim_times(**stim_params)

    A_spikes, C_spikes = generate_neurons(
        stim_times, make_post=True, response=response, **neuron_params)

    B_spikes = generate_neurons(
        stim_times, make_post=False, response=response, **neuron_params)

    beta_AC = causal_connectivity(
        x=A_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    beta_BC = causal_connectivity(
        x=B_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    res = {
        'A_rate': len(A_spikes) / neuron_params['stop_time'],
        'B_rate': len(B_spikes) / neuron_params['stop_time'],
        'C_rate': len(C_spikes) / neuron_params['stop_time'],
        'S_rate': len(stim_times) / neuron_params['stop_time'],
        'C_induced_rate': (len(C_spikes) / neuron_params['stop_time']) - neuron_params['post_rate'],
        'beta_AC': beta_AC,
        'beta_BC': beta_BC
    }
    results_stop_time.append({**stim_params, **neuron_params, **iv_params, **res})

results_stop_time = pd.DataFrame(results_stop_time)
results_stop_time.head()


fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(16, 9))

AC_ground_truth = results_stop_time.C_induced_rate / (results_stop_time.A_rate + results_stop_time.S_rate)
BC_ground_truth = np.zeros(len(results_stop_time['stop_time']))

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    results_stop_time['beta_AC'],
    color='b')

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    results_stop_time['beta_BC'],
    color='r')

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    BC_ground_truth, '--k',
    label='BC ground truth')

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    AC_ground_truth, '--k',
    label='AC ground truth')

ax_e.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    abs(results_stop_time['beta_AC'] - AC_ground_truth),
    label='beta_AC', color='b')

ax_e.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    abs(results_stop_time['beta_BC'] - 0),
    label='beta_BC', color='r')

ax_v.set_xlabel('N trials')
ax_e.set_xlabel('N trials')
ax_e.set_ylabel('Absolute error')
ax_v.set_ylabel('Rate')

ax_v.legend(
    bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
    ncol=2, mode="expand", frameon=False)

ax_e.legend(
    bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
    ncol=2, mode="expand", frameon=False)

plt.show()

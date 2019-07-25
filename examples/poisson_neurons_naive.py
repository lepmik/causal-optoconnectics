import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from causal_optoconnectics import causal_connectivity
from causal_optoconnectics.cch import fit_latency, transfer_probability
from generator import generate_neurons, generate_stim_times
from tqdm import tqdm


np.random.seed(1234)

# response = 'gaussian'
response = 'fixed'

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
trans_prob_params = {
    'y_mu': 4e-3,
    'y_sigma': 3e-3,
    'bin_size': 1e-3,
    'limit': 15e-3,
    'hollow_fraction': .8,
    'width': 100
}

stim_times = generate_stim_times(**stim_params)

A_spikes, C_spikes = generate_neurons(
    stim_times, make_post=True, response=response, **neuron_params)

B_spikes = generate_neurons(
    stim_times, make_post=False, response=response, **neuron_params)

trans_prob_AC,_,_,_,_ = transfer_probability(
    A_spikes, C_spikes, **trans_prob_params)

trans_prob_BC,_,_,_,_ = transfer_probability(
    B_spikes, C_spikes, **trans_prob_params)

print(trans_prob_AC, trans_prob_BC)

plt.figure()
plt.suptitle('Stim times vs A CCH')
fit_latency(stim_times, A_spikes, limit=12e-3, plot=True)

plt.figure()
plt.suptitle('A vs C CCH')
fit_latency(A_spikes, C_spikes, limit=12e-3, plot=True)

plt.figure()
plt.suptitle('B vs C CCH')
fit_latency(B_spikes, C_spikes, limit=12e-3, plot=True)

results_hit_chance = []
for h in tqdm(np.arange(0, 1.1, .1)):
    neuron_params['pre_hit_chance'] = h

    stim_times = generate_stim_times(**stim_params)

    A_spikes, C_spikes = generate_neurons(
        stim_times, make_post=True, response=response, **neuron_params)

    B_spikes = generate_neurons(
        stim_times, make_post=False, response=response, **neuron_params)

    trans_prob_AC,_,_,_,_ = transfer_probability(
        A_spikes, C_spikes, **trans_prob_params)

    trans_prob_BC,_,_,_,_ = transfer_probability(
        B_spikes, C_spikes, **trans_prob_params)

    res = {
        'A_rate': len(A_spikes) / neuron_params['stop_time'],
        'B_rate': len(B_spikes) / neuron_params['stop_time'],
        'C_rate': len(C_spikes) / neuron_params['stop_time'],
        'S_rate': len(stim_times) / neuron_params['stop_time'],
        'C_induced_rate': (len(C_spikes) / neuron_params['stop_time']) - neuron_params['post_rate'],
        'trans_prob_AC': trans_prob_AC,
        'trans_prob_BC': trans_prob_BC
    }
    results_hit_chance.append(
        {**stim_params, **neuron_params, **trans_prob_params, **res})

results_hit_chance = pd.DataFrame(results_hit_chance)

keys = [
    'trans_prob_AC',
    'trans_prob_BC',
]

plt.figure()
cmap = cm.get_cmap('tab10')
cnt = 0
for key in keys:
    plt.plot(results_hit_chance['pre_hit_chance'], results_hit_chance[key], label=key, color=cmap(cnt))
    cnt += 1
plt.grid(True)

AC_ground_truth = results_hit_chance['pre_hit_chance']

BC_ground_truth = np.zeros_like(results_hit_chance['pre_hit_chance'])

plt.plot(results_hit_chance['pre_hit_chance'], AC_ground_truth, '--k')
plt.plot(results_hit_chance['pre_hit_chance'], BC_ground_truth, '--k')

plt.title('Three Poisson neurons with {} response'.format(response))
plt.xlabel('Connection strength')
plt.legend()

neuron_params['pre_hit_chance'] = .8
results_stop_time = []
for h in tqdm(np.arange(100, 6100, 10)):
    stim_params['stop_time'] = h
    neuron_params['stop_time'] = stim_params['stop_time']

    stim_times = generate_stim_times(**stim_params)

    A_spikes, C_spikes = generate_neurons(
        stim_times, make_post=True, response=response, **neuron_params)

    B_spikes = generate_neurons(
        stim_times, make_post=False, response=response, **neuron_params)

    trans_prob_AC = causal_connectivity(
        x=A_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    trans_prob_BC = causal_connectivity(
        x=B_spikes, y=C_spikes, stim_times=stim_times, **iv_params)

    res = {
        'A_rate': len(A_spikes) / neuron_params['stop_time'],
        'B_rate': len(B_spikes) / neuron_params['stop_time'],
        'C_rate': len(C_spikes) / neuron_params['stop_time'],
        'S_rate': len(stim_times) / neuron_params['stop_time'],
        'C_induced_rate': (len(C_spikes) / neuron_params['stop_time']) - neuron_params['post_rate'],
        'trans_prob_AC': trans_prob_AC,
        'trans_prob_BC': trans_prob_BC
    }
    results_stop_time.append({**stim_params, **neuron_params, **iv_params, **res})

results_stop_time = pd.DataFrame(results_stop_time)
results_stop_time.head()

# WARNING this can take some time
fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(16, 9))

AC_ground_truth = np.ones(len(results_stop_time['stop_time'])) * results_hit_chance['pre_hit_chance']
BC_ground_truth = np.zeros(len(results_stop_time['stop_time']))

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    results_stop_time['trans_prob_AC'],
    color='b')

ax_v.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    results_stop_time['trans_prob_BC'],
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
    abs(results_stop_time['trans_prob_AC'] - AC_ground_truth),
    label='trans_prob_AC', color='b')

ax_e.plot(
    results_stop_time['stop_time'] * results_stop_time['S_rate'],
    abs(results_stop_time['trans_prob_BC'] - 0),
    label='trans_prob_BC', color='r')

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

import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib
from functools import partial
import multiprocessing
import click
import re
from causal_optoconnectics.buzsaki import transfer_probability
from causal_optoconnectics.tools import min_error, process_metadata


def load(path):
    data = np.load(path / 'rank_0.npz', allow_pickle=True)
    data = {k: data[k][()] for k in data.keys()}
    X = [np.load(fn, allow_pickle=True)['data'][()] for i, fn in enumerate(path.glob('*.npz')) if i<5]
#     X = [data['data']]
    W_0 = data['W_0']
    W = data['W']
    params = data['params']
    return X, W_0, W, params


def convert_index_to_times(X, params):
    times = np.arange(params['n_time_step']) * params['dt']
    spikes = np.empty((sum([len(x) for x in X]), 2))
    prev_t, prev_len = 0, 0
    for x in X:
        s = x.copy().astype(float)
        s[:,1] = times[x[:,1]]
        s[:,1] = s[:,1] + prev_t
        spikes[prev_len:len(s)+prev_len, :] = s
        prev_t = s[-1,1] + 100
        prev_len += len(s)
    return spikes


def process(pair, W, stim_index, params, spikes):
    source, target = pair
    trans_prob_params = {
        'y_mu': 1.5e-3,
        'y_sigma': 1e-3,
        'bin_size': 1e-3,
        'limit': 2e-2,
        'hollow_fraction': .6,
        'width': 60
    }
    ids = spikes[:, 0]
    pre, post = spikes[ids==source, 1], spikes[ids==target, 1]
    tr = transfer_probability(pre, post, **trans_prob_params)[0]

    result = {
        'source': source,
        'target': target,
        'pair': pair,
        'weight': W[source, target, 0],
        'source_stim': W[stim_index, source, 0] > 0,
        'source_stim_strength': W[stim_index, source, 0],
        'target_stim': W[stim_index, target, 0] > 0,
        'naive_cch': tr
    }
    result.update(params)
    return result


def compute(fn, file_exists, rparams):
    fn_csv = fn / 'naive_cch.csv'
    if fn_csv.with_suffix('.csv').exists():
        if file_exists == 'skip':
            return 
        elif file_exists == 'stop':
            raise OSError(f'File {fn_csv} exists, file_exists={file_exists}')
    X, W_0, W, params = load(fn)
    params.update(rparams)
    stim_index = len(W_0)
    results_meta = pd.DataFrame(process_metadata(
        range(len(W_0)), range(len(W_0)), W=W, stim_index=stim_index))
    sample_meta = results_meta.query(
        f'source_stim and not target_stim and {rparams["target_weight"]}')
    spikes = convert_index_to_times(X, params)
    with multiprocessing.Pool() as p:
        samples = p.map(
            partial(process, W=W, stim_index=stim_index, spikes=spikes, params=params), sample_meta.pair.values)
    samples = pd.DataFrame(samples)
#     samples = pd.DataFrame([process(pair, W=W, stim_index=stim_index, spikes=spikes, params=params) for pair in sample_meta.pair.values])
    save(fn_csv, samples, file_exists)


def save(fname, value, file_exists):
    if fname.exists():
        if file_exists == 'stop':
            raise OSError(f'File exists, file_exists={file_exists}')
        elif file_exists == 'skip':
            return
        elif file_exists == 'new':
            split = fname.stem.split('_')
            if 'version' not in split:
                new_name = fname.stem + '_version_0'
            else:
                split[-1] = str(int(split[-1]) + 1)
                new_name = '_'.join(split)
            fname = fname.with_name(new_name).with_suffix(fname.suffix)
            if fname.exists():
                save(fname, value, file_exists)
        elif file_exists == 'overwrite':
            pass
        else:
            raise ValueError(f'Unknown parameter file_exists={file_exists}')
    if fname.suffix == '.csv':
        value.to_csv(fname)
    elif fname.suffix == '.npz':
        np.savez(fname, value)
    elif fname.suffix == '.yaml':
        yaml = ruamel.yaml.YAML()
        with open(fname, 'w') as f:
            yaml.dump(value, f)
    else:
        raise NotImplementedError(f"Don't know what to do with {fname.suffix}")


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--file-exists', '-f',
              type=click.Choice(['overwrite', 'skip', 'new', 'stop'],
              case_sensitive=False), default='overwrite')
@click.option('--target-weight','-w', default="weight >= 0")
@click.option('--exclude-path-stem','-e', default=(), multiple=True)
def main(data_path, file_exists, target_weight, exclude_path_stem):
    rparams = {
        'target_weight': target_weight,
        'exclude_path_stem': exclude_path_stem
    }
    print("".join([f"{k}: \t{v}\n" for k,v in rparams.items()]))
    data_path = pathlib.Path(data_path).absolute().resolve()
    print(f'Analyzing {data_path}')

    paths = [path for path in data_path.iterdir() if path.is_dir() and path.stem not in exclude_path_stem]
    paths = sorted(paths, key=lambda x: [int(i) for i in re.findall('\\d+', x.stem)])
    data_df = pd.DataFrame({'path': paths})

    iterator = tqdm(data_df.iterrows(), total=len(data_df))
    for i, row in iterator:
        iterator.set_description(row.path.stem)
        sample = compute(row.path, file_exists=file_exists, rparams=rparams)
#     with multiprocessing.Pool() as p:
#         samples = p.map(
#             partial(compute, file_exists=file_exists, rparams=rparams), paths)
        

if __name__ == '__main__':
    main()

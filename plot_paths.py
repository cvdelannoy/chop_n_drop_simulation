import argparse, os, sys
from os.path import splitext
import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from math import inf
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from soma import align
try:
    import pickle5 as pickle
except:
    import pickle
from datetime import datetime
from itertools import chain
import multiprocessing as mp

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)

from helpers import parse_output_dir


def get_sigma(p, res):
    """
    Find std deviation for given threshold and resolution
    """
    return -1 * res / norm.ppf(p * 0.5, 0, 1)


def plot_heatmap(val_mat, path_mat, fp, db_fp, fp_id, db_id):

    pm_bin = np.zeros(path_mat.shape[1:], dtype=int)
    i, j = [x-1 for x in pm_bin.shape]
    pm_bin[-1, -1] = 1
    pp = (i, j)
    # while i > 0 or j > 0:
    while i != -1 or j != -1:
        pm_bin[pp[0], pp[1]] = 1
        # pm_bin[i, j] = 1
        pp = (i, j)
        i, j = path_mat[:, i, j]
    else:
        # pm_bin[i,j] = 1
        pm_bin[pp[0], pp[1]] = 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(pm_bin.astype(float), cbar=False, square=True, annot=False, cmap=sns.color_palette("coolwarm", as_cmap=True),
                center=0.5, vmin=0, vmax=1,
                ax=ax, linewidths=0.5, linecolor='black')
    sns.heatmap(val_mat, cbar=False, square=True, annot=True, alpha=0.0, #linewidths=0.5, linecolor='black',
                annot_kws={'color': 'white', 'fontweight': 'bold'}, # fmt='',
                cmap=ListedColormap(['black']), ax=ax, xticklabels=db_fp.astype(str),
                yticklabels=fp.astype(int).astype(str))
    ax.set_xlabel(f'query: {db_id}'); ax.set_ylabel(f'DB: {fp_id}')
    return fig


def classify_fingerprints(target_dict, db, cdf, algo, soma_cr, sigma, plot_path, save_matching_fps=False):
    rd_out = {}
    max_len_db_fps = max(db)
    matching_fps = {}
    for tid in target_dict:
        fp = target_dict[tid]
        if not len(fp):
            rd_out[tid] = []
            continue

        top3 = [(inf, None)] * 3
        # define search range in number of fragments: 50% shorter/longer, min 1
        fp_len = len(fp)
        d_sr = max(int(fp_len * 0.5), 2)
        min_sr = max(fp_len - d_sr, 1)
        max_sr = max(fp_len + d_sr, 1)
        max_sr = min(max_sr, max_len_db_fps)
        db_range = np.arange(min_sr, max_sr)
        plt_dict = {}
        true_idx = ()
        for nbf in db:
            plt_dict[nbf] = {}
            if nbf not in db_range: continue
            for cidx in db[nbf].index:
                db_fp = db[nbf].loc[cidx, [f'ss{i}' for i in range(nbf)]].to_numpy().astype(np.float)
                score, path = align(fp, db_fp, algo, soma_cr, soma_cr, sigma, True)
                plt_dict[nbf][cidx] = (score, path)
                cdf.loc[(nbf, cidx), 'dtw_score'] = score[-1, -1]
                if tid == db[nbf].loc[cidx, 'seq_id']:
                    true_idx = (nbf, cidx)

        top_idx = cdf.sort_values(['dtw_score'], ascending=True).iloc[:5, :].index
        top_ids = [db[i1].loc[i2, 'seq_id'] for i1, i2, in top_idx]
        if top_ids[0] == tid:
            cur_path = plot_path + 'correct/'
        else:
            cur_path = plot_path + 'incorrect/'
            if len(true_idx):
                score_mat, path_mat = plt_dict[true_idx[0]][true_idx[1]]
                fig = plot_heatmap(score_mat, path_mat,
                                   fp, db[true_idx[0]].loc[
                                       true_idx[1], [f'ss{i}' for i in range(true_idx[0])]].to_numpy().astype(int),
                                   tid, tid)
                fig.savefig(f'{cur_path}{tid}_correct.svg')
                plt.close(fig)
        score_mat, path_mat = plt_dict[top_idx[0][0]][top_idx[0][1]]
        fig = plot_heatmap(score_mat, path_mat,
                           fp, db[top_idx[0][0]].loc[top_idx[0][1], [f'ss{i}' for i in range(top_idx[0][0])]].to_numpy().astype(int),
                           tid, top_ids[0])
        fig.savefig(f'{cur_path}{tid}_vs_{top_ids[0]}.svg')
        plt.close(fig)

        rd_out[tid] = top_ids
        cdf.loc[:, 'dtw_score'] = np.nan

        if save_matching_fps:
            matching_fps[tid] = {f'{top_ids[ii]}_{ii}': db[i1].loc[i2, [f'ss{i}' for i in range(i1)]].to_list() for ii, (i1, i2) in enumerate(top_idx[:3].to_list())}
            matching_fps[tid]['target'] = fp

        # if tid != top_ids[0]:
        #     tt = db[top_idx[0][0]].loc[top_idx[0][1]]
        #     bp = tt.loc[[f'ss{i}' for i in range(top_idx[0][0])]].to_numpy().astype(np.float)
        #     # get real fingerprint
        #     full_db = pd.concat([db[di] for di in db])
        #     rp = full_db.query(f'seq_id == "{tid}"')
        #     tfp = rp[[f'ss{i}' for i in range(len(fp))]].to_numpy().squeeze()
        #     cp=1
    if save_matching_fps:
        return rd_out, matching_fps
    return rd_out


def classify_fingerprints_parallel(target_dict, db, cdf, algo,  soma_cr, sd2, plot_path, save_matching_fps,out_queue):
    out_dict = classify_fingerprints(target_dict, db, cdf, algo, soma_cr, sd2, plot_path, save_matching_fps)
    out_queue.put(out_dict)


parser = argparse.ArgumentParser(description='Return most likely classification for given db and targets, according to dtw score')
parser.add_argument('--db', type=str, required=True)
parser.add_argument('--targets', type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--plot-dir', type=str, required=True)
parser.add_argument('--resolution', type=float, required=True)
parser.add_argument('--save-matching-fps', action='store_true')
parser.add_argument('--soma-cr', type=float, default=4.0)
parser.add_argument('--algorithm', type=str, choices=['dtw', 'soma', 'soma_alt', 'soma_dtw', 'gapped_nw', 'soma_like'], default='soma',
                    help='Define which method to use to determine distance between fingerprints [default:soma]')
parser.add_argument('--cores', type=int, default=4)

args = parser.parse_args()

plot_path = parse_output_dir(args.plot_dir, clean=True)
parse_output_dir(plot_path+'correct')
parse_output_dir(plot_path+'incorrect')

with open(args.db, 'rb') as fh: db = pickle.load(fh)
for nbf in db:
    db[nbf].index.rename('seq_id', inplace=True)
    db[nbf].reset_index(inplace=True)
with open(args.targets, 'rb') as fh: targets = pickle.load(fh)

# Calcualate std used to represent resolution
sd2 = get_sigma(0.5, args.resolution) ** 2

# index database entries
db_id_list = list(chain.from_iterable([[(nb_fragments, idx) for idx in range(len(db[nb_fragments]))] for nb_fragments in db]))
comparison_df = pd.DataFrame(columns=['dtw_score'], index=pd.MultiIndex.from_tuples(db_id_list))

# collect targets
target_dict = {}
for nbf in targets:
    ss_cols = [f'ss{i}' for i in range(nbf)]
    for tid, tup in targets[nbf].iterrows():
        target_dict[tid] = tup.loc[ss_cols].to_numpy()

# compare
if args.cores < 1:
    raise ValueError('provide positive number of cores')
elif args.cores == 1:
    results, matching_fp_dict = classify_fingerprints(target_dict, db, comparison_df.copy(), args.algorithm,
                                                      args.soma_cr, sd2, plot_path, args.save_matching_fps)
else:
    pid_list = np.array_split(np.arange(len(target_dict)), args.cores)
    target_idx_list = list(target_dict)
    target_list = [{target_idx_list[pid]: target_dict[target_idx_list[pid]] for pid in pid_sublist} for pid_sublist in pid_list]
    parallel_results_list = []
    out_queue = mp.Queue()
    processes = [mp.Process(target=classify_fingerprints_parallel,
                            args=(target_list[tidx], db, comparison_df.copy(), args.algorithm, args.soma_cr, sd2,
                                  plot_path, args.save_matching_fps, out_queue))
                 for tidx in range(args.cores)]
    for p in processes:
        p.start()
    while True:
        running = any(p.is_alive() for p in processes)
        while not out_queue.empty():
            parallel_results_list.append(out_queue.get())
        if not running:
            break
    if args.save_matching_fps:
        parallel_results_list, matching_fps = zip(*parallel_results_list)
        matching_fp_dict = {k: v for d in matching_fps for k, v in d.items()}
    results = {k: v for d in parallel_results_list for k, v in d.items()}

# Save results
with open(args.out_pkl, 'wb') as fh: pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
if args.save_matching_fps:
    with open(f'{splitext(args.out_pkl)[0]}_matching_fps.pkl', 'wb') as fh: pickle.dump(matching_fp_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

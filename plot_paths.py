import argparse, os, sys
from os.path import splitext
import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from math import inf
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

import seaborn as sns
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

palette = ['#fbb4ae', '#ccebc5']
palette = ['#fbb4ae', '#4d9221']

def get_sigma(p, res):
    """
    Find std deviation for given threshold and resolution
    """
    return -1 * res / norm.ppf(p * 0.5, 0, 1)


def get_lc_tuples(fp, y_pos):
    sp = 50 # spacing
    fp_cs = np.cumsum(np.concatenate(([0], fp)))
    fp_segments = [[(f1 + sp, y_pos), (f2 - sp, y_pos)] for f1, f2 in zip(fp_cs[:-1], fp_cs[1:])]
    return fp_segments


def plot_heatmap(val_mat, path_mat, fp, db_fp, fp_id, db_id):
    aln_tups = []
    fp_gaps, dbfp_gaps = np.zeros(len(fp), dtype=bool), np.zeros(len(db_fp), dtype=bool)
    pm_bin = np.zeros(path_mat.shape[1:], dtype=int)
    g_list = []
    pm_bin[-1,-1] = 1
    i, j, g = path_mat[:, -1,-1]
    offset = 1
    while i != -1 or j != -1:
        pm_bin[i, j] = 1
        if g == 0:
            aln_tups.extend([((i+offset, 0), (j+offset, 0)), ((i+offset, 1), (j+offset, 1))])
        if g == 1:
            aln_tups.extend([((i+offset, 0), (j+offset, 0)), ((i+offset, 1), (j+1+offset, 1))])
            pm_bin[i+1,j+1] = 1  # 2-to-1 alignment
        elif g == 2:
            fp_gaps[i+1] = True
            aln_tups.extend([((i+offset, 0), (j-1+offset, 1)), ((i+offset, 1), (j-1+offset, 1))])
        elif g == 3:
            dbfp_gaps[j+1] = True
            aln_tups.extend([((i-1+offset, 1), (j+offset, 0)), ((i-1+offset, 1), (j+offset, 1))])

        g_list.append(g)
        i, j, g = path_mat[:, i, j]

    # --- line collection plot ---
    lineplot_lim = 15000
    sp = 100 # spacing
    ls_list = []

    # fingerprint lines
    fp_segments = get_lc_tuples(fp, 1)
    dbfp_segments = get_lc_tuples(db_fp, 0)
    col = palette[1] if db_id == fp_id else palette[0]
    fp_segments_gaps = [s for s, sb in zip(fp_segments, fp_gaps) if sb]
    fp_segments_true = [s for s, sb in zip(fp_segments, fp_gaps) if not sb]
    dbfp_segments_gaps = [s for s, sb in zip(dbfp_segments, dbfp_gaps) if sb]
    dbfp_segments_true = [s for s, sb in zip(dbfp_segments, dbfp_gaps) if not sb]
    ls_list.extend([LineCollection(fp_segments_gaps, colors=len(fp_segments_gaps) * ['black'], alpha=0.5, linewidths=2),
                    LineCollection(fp_segments_true, colors=len(fp_segments_true) * ['black'], alpha=1.0, linewidths=2),
                    LineCollection(dbfp_segments_gaps, colors=len(dbfp_segments_gaps) * [col], alpha=0.5, linewidths=2),
                    LineCollection(dbfp_segments_true, colors=len(dbfp_segments_true) * [col], alpha=1.0, linewidths=2)
                    ])

    # alignment lines
    aln_list = []
    fp_coords = np.vstack((np.cumsum(np.concatenate(([0], fp[:-1]))), np.cumsum(fp))).T
    dbfp_coords = np.vstack((np.cumsum(np.concatenate(([0], db_fp[:-1]))), np.cumsum(db_fp))).T
    for at in aln_tups: aln_list.append([(fp_coords[at[0]], 1), (dbfp_coords[at[1]], 0)])

    ls_list.append(LineCollection(aln_list, colors=len(aln_list) * ['black'],
                                  linewidths=1, linestyles=len(aln_list) * ['--']))

    ln_fig, ax = plt.subplots(1, 1, figsize=(10, 1.5))
    for ls in ls_list: ax.add_collection(ls)
    # ax.autoscale()
    plt.xlim([0,lineplot_lim])
    plt.ylim([-0.2, 1.2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Weight (Da)')
    plt.yticks(list(range(2)), [f'DB: {db_id}', f'Query: {fp_id}'])
    plt.tight_layout()

    # --- matrix plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(pm_bin.astype(float), cbar=False, square=True, annot=False, cmap=sns.color_palette("coolwarm", as_cmap=True),
                center=0.5, vmin=0, vmax=1,
                ax=ax, linewidths=0.5, linecolor='black')
    sns.heatmap(val_mat, cbar=False, square=True, annot=True, alpha=0.0, #linewidths=0.5, linecolor='black',
                annot_kws={'color': 'white', 'fontweight': 'bold'}, # fmt='',
                cmap=ListedColormap(['black']), ax=ax, xticklabels=db_fp.astype(str),
                yticklabels=fp.astype(int).astype(str))
    ax.set_ylabel(f'query: {fp_id}'); ax.set_xlabel(f'DB: {db_id}')
    return fig, ln_fig


def classify_fingerprints(target_dict, db, cdf, algo, soma_cr, sigma, plot_path, save_matching_fps=False):
    rd_out = {}
    max_len_db_fps = max(db)
    matching_fps = {}
    for tid in target_dict:
        fp = target_dict[tid]
        if not len(fp):
            rd_out[tid] = []
            continue

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
                fig, ln_fig = plot_heatmap(score_mat, path_mat,
                                   fp, db[true_idx[0]].loc[
                                       true_idx[1], [f'ss{i}' for i in range(true_idx[0])]].to_numpy().astype(int),
                                   tid, tid)
                fig.savefig(f'{cur_path}{tid}_correct.svg')
                ln_fig.savefig(f'{cur_path}{tid}_correct_lnplot.svg')
                plt.close(fig)
                plt.close(ln_fig)
        score_mat, path_mat = plt_dict[top_idx[0][0]][top_idx[0][1]]
        fig, ln_fig = plot_heatmap(score_mat, path_mat,
                           fp, db[top_idx[0][0]].loc[top_idx[0][1], [f'ss{i}' for i in range(top_idx[0][0])]].to_numpy().astype(int),
                           tid, top_ids[0])
        fig.savefig(f'{cur_path}{tid}_vs_{top_ids[0]}.svg')
        ln_fig.savefig(f'{cur_path}{tid}_vs_{top_ids[0]}_lnfig.svg')
        plt.close(fig)
        plt.close(ln_fig)

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

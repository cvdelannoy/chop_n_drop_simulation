import argparse, os, sys
import numpy as np
import pandas as pd
from tslearn.metrics import dtw, dtw_path
import pickle
from itertools import chain
import multiprocessing as mp


def classify_fingerprints(target_dict, db, cdf):
    rd_out = {}
    for tid in target_dict:
        fp = target_dict[tid]
        for nbf, cidx in cdf.index.to_list():
            cdf.loc[(nbf, cidx), 'dtw_score'] = dtw(fp, db[nbf].loc[cidx, [f'ss{i}' for i in range(nbf)]])
        top_idx = cdf.sort_values(['dtw_score']).iloc[:5, :].index
        top_ids = [db[i1].loc[i2, 'seq_id'] for i1, i2, in top_idx]
        rd_out[tid] = top_ids
    return rd_out

def classify_fingerprints_parallel(target_dict, db, cdf, out_queue):
    out_dict = classify_fingerprints(target_dict, db, cdf)
    out_queue.put(out_dict)


parser = argparse.ArgumentParser(description='Return most likely classification for given db and targets, according to dtw score')
parser.add_argument('--db', type=str, required=True)
parser.add_argument('--targets', type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--cores', type=int, default=4)

args = parser.parse_args()

with open(args.db, 'rb') as fh: db = pickle.load(fh)
for nbf in db:
    db[nbf].index.rename('seq_id', inplace=True)
    db[nbf].reset_index(inplace=True)
with open(args.targets, 'rb') as fh: targets = pickle.load(fh)

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
if args.cores == 1:
    results_dict = classify_fingerprints(target_dict, db, comparison_df.copy())
elif args.cores > 1:
    pid_list = np.array_split(np.arange(len(target_dict)), args.cores)
    target_idx_list = list(target_dict)
    target_list = [{target_idx_list[pid]: target_dict[target_idx_list[pid]] for pid in pid_sublist} for pid_sublist in pid_list]
    parallel_results_list = []
    out_queue = mp.Queue()
    processes = [mp.Process(target=classify_fingerprints_parallel, args=(target_list[tidx], db, comparison_df.copy(), out_queue)) for tidx in range(args.cores)]
    for p in processes:
        p.start()
    while True:
        running = any(p.is_alive() for p in processes)
        while not out_queue.empty():
            parallel_results_list.append(out_queue.get())
        if not running:
            break
    results_dict = {k: v for d in parallel_results_list for k, v in d.items()}

# Save results
with open(args.out_pkl, 'wb') as fh: pickle.dump(results_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

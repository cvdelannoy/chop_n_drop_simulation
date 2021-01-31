import argparse, os, sys
import numpy as np
import pandas as pd
from tslearn.metrics import dtw
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from soma import soma
try:
    import pickle5 as pickle
except:
    import pickle
from datetime import datetime
from itertools import chain
import multiprocessing as mp


def classify_fingerprints(target_dict, db, cdf, algo):
    rd_out = {}
    for tid in target_dict:
        fp = target_dict[tid]
        if not len(fp):
            rd_out[tid] = []
            continue
        for nbf, cidx in cdf.index.to_list():
            if algo == 'soma':
                cdf.loc[(nbf, cidx), 'dtw_score'] = soma(fp, db[nbf].loc[cidx, [f'ss{i}' for i in range(nbf)]].to_numpy().astype(np.float), 4.0, 4.0)
            elif algo == 'dtw':
                cdf.loc[(nbf, cidx), 'dtw_score'] = dtw(fp, db[nbf].loc[cidx, [f'ss{i}' for i in range(nbf)]]) * -1
            else:
                raise ValueError(f'{algo} is not a valid algorithm name')
        top_idx = cdf.sort_values(['dtw_score'], ascending=False).iloc[:5, :].index
        top_ids = [db[i1].loc[i2, 'seq_id'] for i1, i2, in top_idx]
        rd_out[tid] = top_ids
        # if tid != top_ids[0]:
        #     tt = db[top_idx[0][0]].loc[top_idx[0][1]]
        #     bp = tt.loc[[f'ss{i}' for i in range(top_idx[0][0])]].to_numpy().astype(np.float)
        #     # get real fingerprint
        #     full_db = pd.concat([db[di] for di in db])
        #     rp = full_db.query(f'seq_id == "{tid}"')
        #     tfp = rp[[f'ss{i}' for i in range(len(fp))]].to_numpy().squeeze()
        #     cp=1
    return rd_out


def classify_fingerprints_parallel(target_dict, db, cdf, algo, out_queue):
    out_dict = classify_fingerprints(target_dict, db, cdf, algo=algo)
    out_queue.put(out_dict)


parser = argparse.ArgumentParser(description='Return most likely classification for given db and targets, according to dtw score')
parser.add_argument('--db', type=str, required=True)
parser.add_argument('--targets', type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--algorithm', type=str, choices=['dtw', 'soma'], default='soma',
                    help='Define which method to use to determine distance between fingerprints [default:soma]')
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
    results_dict = classify_fingerprints(target_dict, db, comparison_df.copy(), args.algorithm)
elif args.cores > 1:
    pid_list = np.array_split(np.arange(len(target_dict)), args.cores)
    target_idx_list = list(target_dict)
    target_list = [{target_idx_list[pid]: target_dict[target_idx_list[pid]] for pid in pid_sublist} for pid_sublist in pid_list]
    parallel_results_list = []
    out_queue = mp.Queue()
    processes = [mp.Process(target=classify_fingerprints_parallel,
                            args=(target_list[tidx], db, comparison_df.copy(), args.algorithm, out_queue))
                 for tidx in range(args.cores)]
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

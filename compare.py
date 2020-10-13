import argparse, pickle
from os.path import isdir, dirname, abspath
from os import makedirs
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Compare MW fingerprints as stored in a pickled dict of pandas DFs.')
parser.add_argument('--in-pkl',type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--resolution', type=float, default=40000,
                    help='Minimum weight difference that can still be recognized')
parser.add_argument('--scrambled', action='store_true',
                    help='Do not assume sub-sequences are split off in the correct order')
args = parser.parse_args()

with open(args.in_pkl, 'rb') as fh: mw_dict = pickle.load(fh)
out_pkl = abspath(args.out_pkl)
makedirs(dirname(out_pkl), exist_ok=True)

df_list = []
for nb_seqs in mw_dict:
    ss_cols = [f'ss{i}' for i in range(nb_seqs)]
    cdf = mw_dict[nb_seqs].copy()
    if args.scrambled:  # assume we have no information on fragment order --> sort by MW
        cdf.loc[:, ss_cols] = np.sort(cdf.loc[:, ss_cols].to_numpy(), axis=1)
    cdf.loc[:, 'nb_fragments'] = nb_seqs
    cdf.loc[:, 'nb_similar_fingerprints'] = 0
    for ti, tup in cdf.iterrows():
        cdf.loc[ti, 'nb_similar_fingerprints'] = np.sum(np.all((cdf.loc[:, ss_cols] - tup.loc[ss_cols]).abs() < args.resolution, axis=1)) - 1
    cdf.drop(ss_cols, inplace=True, axis=1)
    df_list.append(cdf)
mw_df = pd.concat(df_list)
mw_df.loc[:, 'resolution'] = args.resolution
mw_df.to_pickle(out_pkl)

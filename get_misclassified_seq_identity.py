import argparse, os, sys
from os.path import basename, splitext
import pickle
import numpy as np
from skbio.alignment import global_pairwise_align_protein
from skbio import Protein
from Bio import pairwise2, SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
from helpers import parse_output_dir, parse_input_dir

def load_protein(pid):
    return Protein(str(SeqIO.read(fasta_dict[pid], 'fasta')._seq))


def consensus_fun(tup_list, out_queue):
    out_df = pd.DataFrame(columns=['assigned_id', 'consensus_len'], index=[tup[0] for tup in tup_list])
    for tup in tup_list:
        sid1, sid2, seq1, seq2 = tup
        consensus_frac = np.sum(global_pairwise_align_protein(seq1, seq2)[0].conservation() == 1.0) / len(seq1)
        out_df.loc[sid1, :] = [sid2, consensus_frac]
    out_queue.put(out_df)


parser = argparse.ArgumentParser(description='plot sequence identity for erroneous classifications')
parser.add_argument('--in-pkl', type=str, required=True)
parser.add_argument('--cores', type=int, default=6)
parser.add_argument('--fasta-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
fasta_dict = {basename(splitext(fn)[0]): fn for fn in parse_input_dir(args.fasta_dir, pattern='*.fasta')}

with open(args.in_pkl, 'rb') as fh: fp_dict = pickle.load(fh)

err_id_list = []
for fp_id in fp_dict:
    hit_list = fp_dict[fp_id]
    if len(hit_list):
        if fp_id != hit_list[0]:
            err_id_list.append(fp_id)

# err_id_list = [fp_id for fp_id in fp_dict if fp_dict[fp_id][0] != fp_id]
out_df = pd.DataFrame(columns=['assigned_id', 'consensus_len'], index=err_id_list)
# out_df.loc[:, 'dummy'] = 'incorrect'
nb_err = len(err_id_list)

arg_tups = []

for fpc, fp_id in enumerate(err_id_list):
    query_seq = load_protein(fp_id)
    match_seq = load_protein(fp_dict[fp_id][0])
    arg_tups.append((fp_id, fp_dict[fp_id][0], query_seq, match_seq))

chunk_size = nb_err // args.cores
lims = [0] + [chunk_size + chunk_size * n for n in range(args.cores-1)] + [nb_err]
chunks = [arg_tups[lims[i]:lims[i+1]] for i in range(args.cores)]

out_queue = mp.Queue()
out_list = []
p_list = [mp.Process(target=consensus_fun, args=(at, out_queue)) for at in chunks]
for p in p_list: p.start()
while True:
    running = any(p.is_alive() for p in p_list)
    while not out_queue.empty():
        out_list.append(out_queue.get())
    if not running:
        break
out_df = pd.concat(out_list)
out_df.consensus_len = out_df.consensus_len.astype(float) * 100
out_df = out_df.loc[~out_df.consensus_len.isna(), :]
out_df.to_csv(f'{out_dir}err_identity_distribution.csv')

fig, ax = plt.subplots(figsize=(8.25, 2.9375))
plt.hist(out_df.consensus_len, color='#fbb4ae', bins=20)
plt.xlabel('Identity (%)'); plt.ylabel('# fingerprints')
plt.xlim((0,100))
# sns.violinplot(y='consensus_len', data=out_df)
# sns.stripplot(y='consensus_len', data=out_df)
# sns.histplot('consensus_len', data=out_df)
plt.tight_layout()
plt.savefig(f'{out_dir}err_identity_distribution.svg')


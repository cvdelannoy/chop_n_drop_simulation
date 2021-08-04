import sys, argparse, os
import multiprocessing as mp
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from os.path import basename, splitext
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
from helpers import parse_output_dir, parse_input_dir
from random import choice
import matplotlib.pyplot as plt


def load_protein(pid):
    return str(SeqIO.read(fasta_dict[pid], 'fasta').seq)


def consensus_fun(fid_list, fasta_dict, aligner, out_queue):
    aligner.mode = 'global'
    fasta_list = list(fasta_dict)
    out_df = pd.DataFrame(columns=['assigned_id', 'consensus_len', 'seq1_len'], index=fid_list)
    out_df['consensus_len'] = out_df.consensus_len.astype(float)
    out_df['seq1_len'] = out_df.seq1_len.astype(float)
    for sid1 in fid_list:
        seq1 = load_protein(sid1)
        sid2 = choice(fasta_list)
        seq2 = load_protein(sid2)
        if not len(seq1) or not len(seq2):
            continue
        aln_list = next(aligner.align(seq1, seq2)).aligned[0]
        consensus_frac = np.sum([aln[1] - aln[0] for aln in aln_list]) / len(seq1)
        # consensus_frac = np.sum(global_pairwise_align_protein(seq1, seq2)[0].conservation() == 1.0) / len(seq1)
        out_df.loc[sid1, :] = [sid2, consensus_frac, len(seq1)]
    out_queue.put(out_df)



parser = argparse.ArgumentParser(description='align sequences to random other sequences, record residue identity')
parser.add_argument('--fasta-dir', required=True, type=str)
parser.add_argument('--cores', type=int, default=4)
parser.add_argument('--out-dir', required=True, type=str)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)

fasta_dict = {basename(splitext(fn)[0]): fn for fn in parse_input_dir(args.fasta_dir, pattern='*.fasta')}
fasta_list = list(fasta_dict)
nb_err = len(fasta_dict)

chunk_size = nb_err // args.cores
lims = [0] + [chunk_size + chunk_size * n for n in range(args.cores-1)] + [nb_err]
chunks = [fasta_list[lims[i]:lims[i+1]] for i in range(args.cores)]

out_queue = mp.Queue()
out_list = []
p_list = [mp.Process(target=consensus_fun, args=(at, fasta_dict, PairwiseAligner(), out_queue)) for at in chunks]
for p in p_list: p.start()
while True:
    running = any(p.is_alive() for p in p_list)
    while not out_queue.empty():
        out_list.append(out_queue.get())
    if not running:
        break
out_df = pd.concat(out_list)
out_df.consensus_len = out_df.consensus_len * 100
out_df = out_df.loc[~out_df.consensus_len.isna(), :]
out_df.to_csv(f'{out_dir}err_identity_distribution.csv')

fig, ax = plt.subplots(figsize=(8.25, 2.9375))
plt.hist(out_df.consensus_len, color='#fbb4ae', bins=20)
plt.xlabel('Identity (%)'); plt.ylabel('# sequences')
plt.xlim((0,100))
plt.tight_layout()
plt.savefig(f'{out_dir}err_identity_distribution.svg')

import argparse, os, sys, re, pickle
from os.path import basename, splitext
import pandas as pd
import numpy as np
from helpers import parse_output_dir, parse_input_dir, create_digest_re, digest
from Bio.SeqIO import FastaIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from math import inf

parser = argparse.ArgumentParser(description='In silico digest sequences in fasta files, store their'
                                             'chop-n-drop fingerprints.')
parser.add_argument('--in-dir',type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[0.0, inf])
parser.add_argument('--min-charge', type=float, default=-100)
parser.add_argument('--ph', type=float, default=7.0)
parser.add_argument('--enzyme', type=str, default='trypsin',
                    help='Define which enzyme to use for digestion [default: trypsin]') # todo add options
args = parser.parse_args()

fasta_list = parse_input_dir(args.in_dir, pattern='*.fasta')
nb_proteins = len(fasta_list)
out_pkl = parse_output_dir(os.path.dirname(args.out_pkl), clean=False) + basename(args.out_pkl)
digest_re = create_digest_re(args.enzyme)

mw_dict = {}

print_lim = 1000
for fai, fa_fn in enumerate(fasta_list):
    fa_id = splitext(basename(fa_fn))[0]
    with open(fa_fn) as fh:
        for rec in FastaIO.SimpleFastaParser(fh): header, seq = rec
    if 'X' in seq: continue
    sseq_list = np.array(digest_re.findall(seq))
    sseq_df = pd.DataFrame({'seq': sseq_list,
                            'mw': [ProteinAnalysis(ss).molecular_weight() for ss in sseq_list],
                            'charge': [IsoelectricPoint(ss).charge_at_pH(args.ph) for ss in sseq_list]})
    tot_weight = sseq_df.mw.sum()

    # Remove peptides if too light or too negatively charged, set max weight
    sseq_df.query(f'mw > {args.dynamic_range[0]} and charge > {args.min_charge}', inplace=True)
    sseq_df.loc[sseq_df.mw > args.dynamic_range[1], 'mw'] = args.dynamic_range[1]


    nb_seqs = len(sseq_df)
    entry_dict = {f'ss{i}': ss for i, ss in enumerate(sseq_df.mw)}
    entry_dict['mw'] = tot_weight
    entry_dict['seq_len'] = len(seq)
    if nb_seqs in mw_dict:
        mw_dict[nb_seqs].append(pd.Series(entry_dict, name=fa_id))
    else:
        mw_dict[nb_seqs] = [pd.Series(entry_dict, name=fa_id)]
    if fai > print_lim:
        print(f'{fai+1} of {nb_proteins} digestions done')
        print_lim += 1000

for nb_seqs in mw_dict: mw_dict[nb_seqs] = pd.concat(mw_dict[nb_seqs], axis=1).T
with open(out_pkl, 'wb') as fh:
    pickle.dump(mw_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

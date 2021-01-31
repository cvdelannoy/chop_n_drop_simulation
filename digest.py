import argparse, os, sys, re, pickle
from os.path import basename, splitext
from itertools import chain
import pandas as pd
import numpy as np
from helpers import parse_output_dir, parse_input_dir, create_digest_re, digest
from Bio.SeqIO import FastaIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from math import inf, ceil
from random import random, sample

parser = argparse.ArgumentParser(description='In silico digest sequences in fasta files, store their'
                                             'chop-n-drop fingerprints.')
parser.add_argument('--in-dir',type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[0.0, inf],
                    help='Dynamic range of measurement method. mw lower than low is ignored, mw higher than high is'
                         'set to high. [default: 0.0 inf]')
parser.add_argument('--min-charge', type=float, default=-100,
                    help='Lowest charge allowed for a fragment. Lower-charge fragments are ignored. [default -100]')
parser.add_argument('--ph', type=float, default=7.0,
                    help='pH, influence particle charge [default: 7.0]')
parser.add_argument('--resolution', type=float, default=0.0,
                    help='Minimum weight difference that can still be recognized. Implemented here as 1 sd of the'
                         'gaussian noise added. [default: 0.0]')
parser.add_argument('--efficiency', type=float, default=1.0,
                    help='Define what fraction of sites is actually cleaved. [default: 1.0]')
parser.add_argument('--specificity', type=float, default=1.0,
                    help='1 - probability that cleavage occurs after any residue at random. [default: 1.0]')  # todo: implementation not done
parser.add_argument('--catch-rate', type=float, default=1.0,
                    help='Fraction of fragments that is registered, at random [default: 1.0]')
parser.add_argument('--repeats', type=int, default=1,
                    help='Number of times to repeat digestion. Note: only makes sense if efficiency/specificity/resolution is set, '
                         'otherwise fingerprints will end up the same each time. [default: 1]')
parser.add_argument('--enzyme', type=str, default='trypsin',
                    help='Define which enzyme to use for digestion [default: trypsin]') # todo add options
parser.add_argument('--subsampling-fraction', default=1.0, type=float,
                    help='When subsampling sequences for targets db, define what fraction is taken [default: 1.0]')
args = parser.parse_args()

fasta_list = parse_input_dir(args.in_dir, pattern='*.fasta')
if args.subsampling_fraction < 1.0:
    nb_subsampled = ceil(len(fasta_list) * args.subsampling_fraction)
    fasta_list = sample(fasta_list, nb_subsampled)
nb_proteins = len(fasta_list)
out_pkl = parse_output_dir(os.path.dirname(args.out_pkl), clean=False) + basename(args.out_pkl)
digest_re = create_digest_re(args.enzyme)

mw_dict = {}

print_lim = 1000
for fai, fa_fn in enumerate(fasta_list):
    for nr in range(args.repeats):
        fa_id = splitext(basename(fa_fn))[0]
        with open(fa_fn) as fh:
            for rec in FastaIO.SimpleFastaParser(fh): header, seq = rec
        if 'X' in seq: continue

        # Digest
        sseq_list = np.array(digest_re.findall(seq))

        # if args.specificity < 1.0:
        #     # specificity: pre-digest at random
        #     cleave_idx = np.concatenate(([0], np.argwhere(np.random.random(len(seq)-2) > args.specificity).squeeze(-1) + 1, [len(seq)]))
        #     seqs_list = [seq[cleave_idx[ii-1]:i] for ii, i in enumerate(cleave_idx)][1:]
        #     sseq_list = list(chain.from_iterable([np.array(digest_re.findall(ss)) for ss in seqs_list]))

        sseq_list_copy = sseq_list.copy()

        # Fuse fractions at random
        if args.efficiency < 1.0 and len(sseq_list) > 1:
            sseq_fusions = [sseq_list[0]]
            for ss in sseq_list[1:]:
                if random() > args.efficiency:
                    sseq_fusions[-1] += ss
                else:
                    sseq_fusions.append(ss)
            sseq_list = sseq_fusions

        sseq_df = pd.DataFrame({'seq': sseq_list,
                                'mw': [ProteinAnalysis(ss).molecular_weight() for ss in sseq_list],
                                'charge': [IsoelectricPoint(ss).charge_at_pH(args.ph) for ss in sseq_list]})
        tot_weight = sseq_df.mw.sum()

        # Add gaussian noise
        sseq_df.mw += np.random.normal(0, args.resolution, len(sseq_df))

        # Remove peptides if too light or too negatively charged, set max weight
        sseq_df.query(f'mw > {args.dynamic_range[0]}', inplace=True)
        # sseq_df.query(f'mw > {args.dynamic_range[0]} and charge > {args.min_charge}', inplace=True)  # todo test
        sseq_df.loc[sseq_df.mw > args.dynamic_range[1], 'mw'] = args.dynamic_range[1]

        # Apply random catch rate
        sseq_df = sseq_df.iloc[np.random.rand(len(sseq_df)) < args.catch_rate]

        nb_seqs = len(sseq_df)
        entry_dict = {f'ss{i}': ss for i, ss in enumerate(sseq_df.mw)}
        entry_dict['mw'] = tot_weight
        entry_dict['seq_len'] = len(seq)
        if nb_seqs in mw_dict:
            mw_dict[nb_seqs].append(pd.Series(entry_dict, name=fa_id))
        else:
            mw_dict[nb_seqs] = [pd.Series(entry_dict, name=fa_id)]

for nb_seqs in mw_dict: mw_dict[nb_seqs] = pd.concat(mw_dict[nb_seqs], axis=1).T
with open(out_pkl, 'wb') as fh:
    pickle.dump(mw_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

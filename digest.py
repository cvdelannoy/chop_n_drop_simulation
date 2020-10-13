import argparse, os, sys, re, pickle
from os.path import basename, splitext
import pandas as pd
from helpers import parse_output_dir, parse_input_dir, create_digest_re, digest
from Bio.SeqIO import FastaIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


parser = argparse.ArgumentParser(description='In silico digest sequences in fasta files, store their'
                                             'chop-n-drop fingerprints.')
parser.add_argument('--in-dir',type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)
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
    sseq_list = digest_re.findall(seq)
    mw_list = [ProteinAnalysis(ss).molecular_weight() for ss in sseq_list]
    nb_seqs = len(sseq_list)
    entry_dict = {f'ss{i}': ss for i, ss in enumerate(mw_list)}
    entry_dict['mw'] = sum(mw_list)
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

import argparse
import pandas as pd
from os import rename
from os.path import basename
import numpy as np
import requests
from Bio import SeqIO
from math import inf

from helpers import parse_input_dir, parse_output_dir, print_timestamp

parser = argparse.ArgumentParser(description='Query UniProt for potential biomarkers for  given disease, '
                                    'with max number of given residues, etc.')
key_group = parser.add_mutually_exclusive_group()
key_group.add_argument('--disease', type=str,
                       help='Disease to query for.')
key_group.add_argument('--keyword', type=str,
                       help='UniProt keyword to filter on. Generally returns broader results, see '
                            'https://www.uniprot.org/keywords/ for an overview.')
key_group.add_argument('--type', type=str,
                       help='UniProt type to filter on. more flexible than disease, see '
                            'https://www.uniprot.org/keywords/ for an overview.')
parser.add_argument('--include-unreviewed', action='store_true',
                    help='Also include not manually curated entries (i.e. outside Swiss-Prot).')
parser.add_argument('--res-frequencies', type=str, nargs='+', default=[],
                    help='Frequency of residues to look for, in format: [1_LETTER_CODE][MIN]-[MAX], e.g.:'
                         'K2-10 C1-5 means 2 to 5 lysines OR 1 to 5 cysteines')
parser.add_argument('--len-range', type=str, nargs=2, default= (str(0), 'inf'),
                    help='min and max length of sequence in integers ("inf" for no upper limit), e.g.: 20 100')
parser.add_argument('--out-dir', type=str, required=True,
                    help='Directory where results are stored.')
args = parser.parse_args()

# --- argument parsing ---
out_dir = parse_output_dir(args.out_dir, clean=True)
raw_fasta_dir = parse_output_dir(out_dir+'raw_fasta')
fasta_dir = parse_output_dir(out_dir+'fasta')

res_freqs = {}
for rf in args.res_frequencies:
    try:
        freqs_str = rf[1:]
        if freqs_str[0] == '-':
            res_min = 0; res_max = int(freqs_str[1:])
        elif freqs_str[-1] == '-':
            res_max = inf; res_min = int(freqs_str[:-1])
        else:
            res_min, res_max = [int(fr) for fr in rf[1:].split('-')]
    except:
        raise ValueError(f'Residue frequency argument could not be parsed: {rf}')
    if res_min >= res_max:
        raise ValueError(f'Residue frequency error for {rf}: minimum must be below maximum')
    res_freqs[rf[0]] = {'min': res_min, 'max': res_max}

len_range = [int(args.len_range[0])]
if args.len_range[1] == 'inf':
    len_range.append(inf)
else:
    len_range.append(int(args.len_range[1]))
if len_range[1] < len_range[0]:
    raise ValueError('In supplied length range, minimum must be smaller than maximum')

# --- query uniprot db on disease ---
query = f'proteome:UP000005640 '
if not args.include_unreviewed: query += 'reviewed:yes '
if args.include_unreviewed:
    query += ''
if args.disease:
    query += f'annotation:(type:disease "{args.disease}")'
elif args.keyword:
    query = f'keyword: "{args.keyword}"'
elif args.type:
    query = f'annotation:(type:{args.type})'
request_params={
    'query': query,
    'include': 'yes',
    'format': 'fasta'
}

req_obj = requests.get("http://www.uniprot.org/uniprot/", params=request_params)
if not req_obj.ok:
    raise ValueError(f'Request to uniprot db failed, status: {req_obj.status_code}')

for fit, fasta_str in enumerate(req_obj.content.decode('utf-8').split('>')[1:]):
    try:
        if '|' in fasta_str:
            fasta_fn = fasta_str.split('|')[1]
        else:
            fasta_fn = f'protein_name_unparsable_{fit}'
    except:
        print(f'parsing problem encountered for {fasta_str}')
        continue
    with open(f'{raw_fasta_dir}{fasta_fn}.fasta', 'w') as fh:
        fh.write('>' + fasta_str)

# --- filter on sequence ---
df_columns = list(res_freqs) + ['seq_length', 'description']
out_df = pd.DataFrame(columns=df_columns)
fastas = parse_input_dir(f'{raw_fasta_dir}', pattern='*.fasta')
nb_fastas = len(fastas)
print_idx = 99
for pidx, fasta in enumerate(fastas):
    seq_obj = list(SeqIO.parse(fasta, 'fasta'))[0]
    seq = np.array(seq_obj.seq)
    if not len_range[0] <= seq.size <= len_range[1]: continue
    if len(res_freqs):
        rf_check = [res_freqs[res]['min'] <= np.sum(seq == res) <= res_freqs[res]['max'] for res in res_freqs]
        if not any(rf_check): continue
    patch = {res: np.sum(seq == res) for res in res_freqs}
    patch['seq_length'] = seq.size
    patch['description'] = seq_obj.description
    out_df.loc[seq_obj.id] = patch
    rename(fasta, fasta_dir + basename(fasta))
    if pidx >= print_idx:
        print(f'{print_timestamp()} {pidx + 1} of {nb_fastas} proteins treated')
        print_idx += 100
out_df.to_csv(f'{out_dir}hits.tsv', sep='\t')
print(f'{print_timestamp()} Done, hits table in {out_dir}hits.tsv, fastas in {out_dir}fastas')

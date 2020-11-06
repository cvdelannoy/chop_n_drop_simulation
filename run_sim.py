import os, sys
import argparse
from jinja2 import Template
from helpers import parse_output_dir
import snakemake as sm
from math import inf
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


parser = argparse.ArgumentParser(description='Run full chop n drop simulation.')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--fasta-dir', type=str, required=False)
parser.add_argument('--enzyme', type=str, default='trypsin',
                    help='Enzyme with which to digest. [default: trypsin]')
parser.add_argument('--res-range', nargs=2, type=float, default=[1, 21],
                    help='Weight resolution range to visit with 20 increments. [default: 1 21]')
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[0.0, 10E99],
                    help='Lower and upper limit between which resolution is valid. Below lower limit,'
                         'particle is not registered, above higher limit it is registered at maximum weight.'
                         '[default: 0 10E99]')
parser.add_argument('--min-charge', type=float, default=-100,
                    help='Minimum charge in units at which fragments are still passing the pore. [default: -100]')
parser.add_argument('--ph', type=float, default=7.0,
                    help='pH at which charge of fragment is calculated. [default: 7.0]')
parser.add_argument('--scrambled', action='store_true',
                    help='Ignore order of fragments in constructing fingerprint.')
parser.add_argument('--cores', type=int, default=4,
                    help='Max number of cores to engage. Only of importance during comparison step. [default: 4]')
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)

with open(f'{__location__}/chop_n_drop_sim.sf', 'r') as fh: sf_template_txt = fh.read()
sf_txt = Template(sf_template_txt, ).render(
    __location__=__location__,
    out_dir=out_dir,
    enzyme=args.enzyme,
    fasta_dir=args.fasta_dir,
    scrambled=args.scrambled,
    min_mw=args.dynamic_range[0], max_mw=args.dynamic_range[1],
    min_charge=args.min_charge, ph=args.ph,
    min_res=args.res_range[0], max_res=args.res_range[1]
)

sf_fn = f'{out_dir}chop_n_drop_pipeline.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)

sm.snakemake(sf_fn, cores=args.cores)

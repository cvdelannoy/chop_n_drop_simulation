import os, sys
from shutil import copyfile
import argparse
from jinja2 import Template
from helpers import parse_output_dir
import snakemake as sm
from math import inf
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


parser = argparse.ArgumentParser(description='Run full chop n drop simulation.')

# --- IO ---
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--fasta-dir', type=str, required=False)
parser.add_argument('--db', type=str, required=False,
                    help='Use provided pickled db instead of creating one with given fastas')

# --- enzyme properties ---
parser.add_argument('--enzyme', type=str, default='trypsin',
                    help='Enzyme with which to digest. [default: trypsin]')
parser.add_argument('--efficiency', type=float, default=1.0,
                    help='Cleaving efficiency of enzyme. [default: 1.0]')
parser.add_argument('--efficiency-range', type=float, nargs=2, default=[0.5, 1.0],
                    help='Lower and upper limit of efficiencies if a ranged mode is used [default: 0.5, 1.0]')
parser.add_argument('--specificity', type=float, default=1.0,
                    help='Cleaving specificity of enzyme, 1 - the probability of cleaving at a random position [default: 1.0]')

# --- pore properties ---
parser.add_argument('--resolution', type=float, default=4,
                    help='Weight resolution')
parser.add_argument('--res-range', nargs=2, type=float, default=[1, 21],
                    help='Weight resolution range to visit with 20 increments. For uniqueness mode [default: 1 21]')
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[0.0, 10E99],
                    help='Lower and upper limit between which resolution is valid. Below lower limit,'
                         'particle is not registered, above higher limit it is registered at maximum weight.'
                         '[default: 0 10E99]')
parser.add_argument('--nb-range-steps', type=int, default=5,
                    help='Number of steps to make in allowed range for res, catch rate and efficiency')
parser.add_argument('--min-charge', type=float, default=-100,
                    help='Minimum charge in units at which fragments are still passing the pore. [default: -100]')
parser.add_argument('--catch-rate', type=float, default=1.0,
                    help='Alternative to ph/min charge; define what fraction of fragments are caught [default: 1.0]')
parser.add_argument('--cr-range', type=float, nargs=2, default=[0.8, 1.0],
                    help='Range between which to pick catch rates [default: 0.8 1.0].')

# --- target(altering) properties ---
parser.add_argument('--ph', type=float, default=7.0,
                    help='pH at which charge of fragment is calculated. [default: 7.0]')
parser.add_argument('--scrambled', action='store_true',
                    help='Ignore order of fragments in constructing fingerprint.')
parser.add_argument('--repeats', type=int, default=20,
                    help='Number of times digestion is repeated. [default: 20]')
parser.add_argument('--subsampling-fraction', default=0.01, type=float,
                    help='When subsampling sequences for targets db, define what fraction is taken [default: 0.01]')

# --- misc ---
parser.add_argument('--algorithm', type=str, choices=['dtw', 'soma'], default='soma',
                    help='Define which method to use to determine distance between fingerprints [default:soma]')
parser.add_argument('--cores', type=int, default=4,
                    help='Max number of cores to engage simultaneously. [default: 4]')
parser.add_argument('--mode', type=str, choices=['uniqueness', 'perfect_db', 'unknown_sample', 'perfect_db_range'],
                    default='perfect_db',
                    help='Type of analysis to perform, must be one of the following: '
                         '[uniqueness]: assume perfect fingerprints and assess whether fingerprints are unique'
                         '[perfect_db]: error-less digestion for comparison database, errors according to parameters for test data'
                         '[unknown_sample]: repeat db generation several times with errors, compare test fingerprints against it')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
if args.db:
    copyfile(args.db, f'{out_dir}digested_products_database.pkl')

if args.mode == 'uniqueness':
    # Option 1: simulate fingerprints once and evaluate uniqueness
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
elif args.mode == 'unknown_sample':
    # Option 2: simulate repeated digestions for nb_repeats times, classification and evaluation
    with open(f'{__location__}/chop_n_drop_sim_with_classification.sf', 'r') as fh: sf_template_txt = fh.read()
    sf_txt = Template(sf_template_txt, ).render(
        __location__=__location__,
        out_dir=out_dir,
        enzyme=args.enzyme,
        fasta_dir=args.fasta_dir,
        scrambled=args.scrambled,
        min_mw=args.dynamic_range[0], max_mw=args.dynamic_range[1],
        min_charge=args.min_charge, ph=args.ph,
        resolution=args.resolution,
        efficiency=args.efficiency, specificity=args.specificity, repeats=args.repeats,
        cores=args.cores,
        algo=args.algorithm
    )
elif args.mode == 'perfect_db':
    # Option 3: simulate perfect fingerprints once for db, then classification/evaluation against that
    with open(f'{__location__}/chop_n_drop_sim_with_classification_v2.sf', 'r') as fh: sf_template_txt = fh.read()
    sf_txt = Template(sf_template_txt, ).render(
        __location__=__location__,
        out_dir=out_dir,
        enzyme=args.enzyme,
        fasta_dir=args.fasta_dir,
        scrambled=args.scrambled,
        min_mw=args.dynamic_range[0], max_mw=args.dynamic_range[1],
        min_charge=args.min_charge, ph=args.ph,
        resolution=args.resolution,
        efficiency=args.efficiency, specificity=args.specificity,
        catch_rate=args.catch_rate,
        cores=args.cores,
        algorithm=args.algorithm
    )
elif args.mode == 'perfect_db_range':
    # Option 4: as option 3, but run for range of resolutions and efficiencies
    with open(f'{__location__}/chop_n_drop_sim_with_classification_v2_ranges.sf', 'r') as fh: sf_template_txt = fh.read()
    sf_txt = Template(sf_template_txt, ).render(
        __location__=__location__,
        out_dir=out_dir,
        enzyme=args.enzyme,
        fasta_dir=args.fasta_dir,
        repeats=args.repeats,
        subsampling_fraction=args.subsampling_fraction,
        nb_range_steps=args.nb_range_steps,
        scrambled=args.scrambled,
        min_mw=args.dynamic_range[0], max_mw=args.dynamic_range[1],
        min_cr=args.cr_range[0], max_cr=args.cr_range[1],
        min_charge=args.min_charge, ph=args.ph,
        resolution=args.resolution,
        efficiency=args.efficiency, specificity=args.specificity,
        catch_rate=args.catch_rate,
        cores=args.cores,
        algorithm=args.algorithm,
        min_res=args.res_range[0], max_res=args.res_range[1],
        min_eff=args.efficiency_range[0], max_eff=args.efficiency_range[1]
    )
else:
    raise ValueError(f'--mode option "{args.mode}" is unknown')

sf_fn = f'{out_dir}chop_n_drop_pipeline.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)

sm.snakemake(sf_fn, cores=args.cores, dryrun=args.dry_run, verbose=False)

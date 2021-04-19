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
parser.add_argument('--efficiency', type=float, default=0.99,
                    help='Cleaving efficiency of enzyme. [default: 1.0]')
parser.add_argument('--efficiency-range', type=float, nargs='+', default=[0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    help='range of efficiencies if a ranged mode is used [default 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]')
parser.add_argument('--specificity', type=float, default=1.0,
                    help='Cleaving specificity of enzyme, 1 - the probability of cleaving at a random position [default: 1.0]')

# --- pore properties ---
parser.add_argument('--resolution', type=float, default=4,
                    help='Weight resolution')
parser.add_argument('--res-range', nargs='+', type=float, default=[4, 100, 200, 300, 400, 500],
                    help='Weight resolution range to visit [default: 4, 100, 200, 300, 400, 500]')
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[500.0, 2000.0],
                    help='Lower and upper limit between which resolution is valid. Below lower limit,'
                         'particle is not registered, above higher limit it is registered at maximum weight.'
                         '[default: 500 2000]')
parser.add_argument('--nb-range-steps', type=int, default=6,
                    help='Number of steps to make in allowed range for res, catch rate and efficiency [default: 6]')
parser.add_argument('--min-charge', type=float, default=-100,
                    help='Minimum charge in units at which fragments are still passing the pore. [default: -100]')
parser.add_argument('--catch-rate', type=float, default=0.99,
                    help='Alternative to ph/min charge; define what fraction of fragments are caught [default: 1.0]')
parser.add_argument('--cr-range', type=float, nargs='+', default=[0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    help='Range between which to pick catch rates [default: 0.75 1.0].')
parser.add_argument('--soma-cr', type=float, default=4.0)
parser.add_argument('--soma-cr-range', type=float, nargs='+', default=[4.0, 10.0, 100.0])

# --- target(altering) properties ---
parser.add_argument('--ph', type=float, default=7.0,
                    help='pH at which charge of fragment is calculated. [default: 7.0]')
parser.add_argument('--scrambled', action='store_true',
                    help='Ignore order of fragments in constructing fingerprint.')
parser.add_argument('--repeats', type=int, default=5,
                    help='Number of times digestion is repeated. [default: 5]')
parser.add_argument('--subsampling-fraction', default=1.0, type=float,
                    help='When subsampling sequences for targets db, define what fraction is taken [default: 1.0]')

# --- misc ---
parser.add_argument('--algorithm', type=str, choices=['dtw', 'soma', 'soma_alt', 'gapped_nw', 'soma_like'], default='soma',
                    help='Define which method to use to determine distance between fingerprints [default:soma]')
parser.add_argument('--cores', type=int, default=4,
                    help='Max number of cores to engage simultaneously. [default: 4]')
parser.add_argument('--mode', type=str, choices=['uniqueness', 'perfect_db', 'unknown_sample', 'perfect_db_range'],
                    default='perfect_db',
                    help='Type of analysis to perform, must be one of the following: '
                         '[uniqueness]: assume perfect fingerprints and assess whether fingerprints are unique'
                         '[perfect_db]: error-less digestion for comparison database, errors according to parameters for test data'
                         '[unknown_sample]: repeat db generation several times with errors, compare test fingerprints against it'
                         '[perfect_db_range]: repeat pefect_db several times, over range of catch rate/resolution/efficiency values')
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
elif args.mode in ('perfect_db', 'perfect_db_range'):
    if args.mode == 'perfect_db':
        target_rule = f'''
rule target:
    input:
        f'{out_dir}resolution{args.resolution}_catch_rate{args.catch_rate}_efficiency{args.efficiency}_somacr{args.soma_cr}/plots/0/classification_eval.csv'
        '''
    elif args.mode == 'perfect_db_range':
        # Option 4: as option 3, but run for range of resolutions, efficiencies and soma_cr values
        target_rule = '''
rule target:
    input:
        res_svg=f'{out_dir}meta_plots/accuracy_vs_resolution.svg',
        cr_svg=f'{out_dir}meta_plots/accuracy_vs_catch_rate.svg',
        eff_svg=f'{out_dir}meta_plots/accuracy_vs_efficiency.svg',
        somacr_svg=f'{out_dir}meta_plots/accuracy_vs_somacr.svg'
    '''

    with open(f'{__location__}/chop_n_drop_sim_with_classification_v2_ranges.sf', 'r') as fh: sf_template_txt = fh.read()
    sf_txt = Template(sf_template_txt, ).render(
        __location__=__location__,
        target_rule=target_rule,
        out_dir=out_dir,
        enzyme=args.enzyme,
        fasta_dir=args.fasta_dir,
        repeats=args.repeats,
        subsampling_fraction=args.subsampling_fraction,
        nb_range_steps=args.nb_range_steps,
        scrambled=args.scrambled,
        min_mw=args.dynamic_range[0], max_mw=args.dynamic_range[1],
        min_charge=args.min_charge, ph=args.ph,
        resolution=args.resolution,
        efficiency=args.efficiency, specificity=args.specificity,
        catch_rate=args.catch_rate,
        cores=args.cores,
        algorithm=args.algorithm,
        soma_cr=args.soma_cr,
        res_range=args.res_range, eff_range=args.efficiency_range, cr_range=args.cr_range, soma_cr_range=args.soma_cr_range
    )
else:
    raise ValueError(f'--mode option "{args.mode}" is unknown')

sf_fn = f'{out_dir}chop_n_drop_pipeline.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)

sm.snakemake(sf_fn, cores=args.cores, dryrun=args.dry_run, verbose=False)

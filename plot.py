import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from helpers import parse_output_dir

def plot_barhist(cdf, feat, top_pct=None, binwidth=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    if top_pct:
        cdf = cdf.sort_values([feat]).head(int(len(cdf) * top_pct))
    sns.histplot(cdf, x=feat, hue='similar_cls', stat='probability', multiple='fill',
                 hue_order=lab_order, binwidth=binwidth, element='step', linewidth=0, palette=['red', 'yellow', 'blue'],
                 ax=ax)
    return fig



parser = argparse.ArgumentParser(description='Plot chop n drop simulation comparison results of several dicts.')
parser.add_argument('--in-pkl', nargs='+', type=str)
parser.add_argument('--top-n', type=int, default=3,
                    help='Report unique, <= n similiar and > n similar fingerprints. [default: 3]')
parser.add_argument('--out-dir', type=str)

args = parser.parse_args()
out_dir = parse_output_dir(args.out_dir)

compare_df = pd.concat([pd.read_pickle(fn) for fn in args.in_pkl])

compare_df.loc[:, 'similar_cls'] = f'more than {args.top_n}'
compare_df.loc[compare_df.nb_similar_fingerprints < args.top_n, 'similar_cls'] = f'up to {args.top_n}'
compare_df.loc[compare_df.nb_similar_fingerprints == 0, 'similar_cls'] = 'unique'
lab_order = [f'more than {args.top_n}', f'up to {args.top_n}', 'unique']

for res, cdf in compare_df.groupby('resolution'):
    # --- nb fragments ---
    fig = plot_barhist(cdf, 'nb_fragments')
    fig.gca().set_xlabel('# fragments'); fig.gca().set_ylabel('fraction')
    plt.savefig(f'{out_dir}uniqueness_vs_nb_fragments_res{res}_all.svg', dpi=400)
    plt.close(fig)

    fig = plot_barhist(cdf, 'nb_fragments', top_pct=0.95)
    fig.gca().set_xlabel('# fragments'); fig.gca().set_ylabel('fraction')
    plt.savefig(f'{out_dir}uniqueness_vs_nb_fragments_res{res}_top95.svg', dpi=400)
    plt.close(fig)

    # --- sequence length ---
    fig = plot_barhist(cdf, 'seq_len')
    fig.gca().set_xlabel('sequence length'); fig.gca().set_ylabel('fraction')
    plt.savefig(f'{out_dir}uniqueness_vs_seqlen_res{res}_all.svg', dpi=400)
    plt.close(fig)

    fig = plot_barhist(cdf, 'seq_len', top_pct=0.95)
    fig.gca().set_xlabel('sequence length'); fig.gca().set_ylabel('fraction')
    plt.savefig(f'{out_dir}uniqueness_vs_seqlen_res{res}_top95.svg', dpi=400)
    plt.close(fig)

# --- resolution ---
fig = plot_barhist(compare_df, 'resolution')
fig.gca().set_xlabel('resolution'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}uniqueness_vs_resolution.svg', dpi=400)
plt.close(fig)

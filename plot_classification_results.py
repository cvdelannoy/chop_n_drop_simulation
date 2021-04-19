import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
try:
    import pickle5 as pickle
except:
    import pickle

from helpers import parse_output_dir

# lab_order = ['misclassified', 'top3', 'top1']
# palette = ['#fbb4ae', '#fed9a6', '#ccebc5']

lab_order = ['misclassified', 'correct']
palette = ['#fbb4ae', '#ccebc5']


def plot_barhist(cdf, feat, top_pct=None, binwidth=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    if top_pct:
        cdf = cdf.sort_values([feat]).head(int(len(cdf) * top_pct))
    sns.histplot(cdf, x=feat, hue='pred', stat='probability', multiple='fill',
                 hue_order=lab_order, binwidth=binwidth, element='step', linewidth=0, palette=palette,
                 ax=ax)
    return fig

def plot_abs_barhist(cdf, feat, feat_str, feat_range):
    wdim = 15
    fig = plt.figure(figsize=(8.25, 2.9375))
    gs = gridspec.GridSpec(1, wdim, figure=fig)

    # Plot main histogram
    ax = fig.add_subplot(gs[0, :wdim - 1])
    target_dict = {cl: sdf.loc[:, feat].to_numpy() for cl, sdf in cdf.groupby('pred')}
    target_dict = {cl: target_dict[cl] for cl in ('correct', 'misclassified')}
    ax.hist(target_dict.values(), 30, stacked=True, range=feat_range, color=palette[::-1])
    plt.xlabel(feat_str); plt.ylabel('# proteins'); plt.xlim(*feat_range)

    # plot sidebar
    ax = fig.add_subplot(gs[0, wdim - 1:])
    cdf.loc[:, 'dummy'] = 0
    sns.histplot(cdf, x='dummy', hue='pred', stat='probability', multiple='fill',
                 hue_order=['misclassified', 'correct'], linewidth=0, palette=palette, ax=ax)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.xlabel('')
    plt.ylabel('fraction')
    plt.tight_layout()
    ax.get_legend().remove()
    return fig


parser = argparse.ArgumentParser(description='Ṕlot output of classify.py')
parser.add_argument('--classify-pkl', type=str, required=True)
parser.add_argument('--targets-pkl', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)

args = parser.parse_args()

with open(args.classify_pkl, 'rb') as fh: classify_dict = pickle.load(fh)
with open(args.targets_pkl, 'rb') as fh: target_dict = pickle.load(fh)

out_dir = parse_output_dir(args.out_dir)

for nfrag in target_dict:
    target_dict[nfrag].drop([f'ss{i}' for i in range(nfrag)], inplace=True, axis=1)
    target_dict[nfrag].loc[:, 'nb_fragments'] = nfrag
target_df = pd.concat(target_dict.values())

result_df = pd.DataFrame(index=list(classify_dict), columns=['mw', 'seq_len', 'nb_fragments', 'pred'])
for pid in classify_dict:
    if not len(classify_dict[pid]): pred = 'misclassified'
    elif classify_dict[pid][0] == pid: pred = 'correct'
    # elif pid in classify_dict[pid][:3]: pred = 'top3'
    else: pred = 'misclassified'
    result_df.loc[pid, :] = target_df.loc[pid].to_list() + [pred]
result_df.to_csv(f'{out_dir}classification_eval.csv')
summary_df = pd.DataFrame({'correct': [len(result_df.query('pred == "correct"')) / len(result_df)],
                           # 'top3': [len(result_df.query('pred != "misclassified"')) / len(result_df)]
                           }, index=['accuracy'])
summary_df.to_csv(f'{out_dir}accuracy_summary.csv')

# --- plot entire set ---

fig = plot_abs_barhist(result_df, 'seq_len', 'sequence length', (1, 1500))
plt.savefig(f'{out_dir}seqlen_vs_cls_abs.svg', dpi=400)
plt.close(fig)

fig = plot_abs_barhist(result_df, 'nb_fragments', '# fragments', (1, 50))
plt.savefig(f'{out_dir}nfrag_vs_cls_abs.svg', dpi=400)
plt.close(fig)

fig = plot_abs_barhist(result_df, 'mw', 'weight (Da)', (0, 50000))
plt.savefig(f'{out_dir}mw_vs_cls_abs.svg', dpi=400)
plt.close(fig)

fig = plot_barhist(result_df, 'seq_len')
fig.gca().set_xlabel('sequence length'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}seqlen_vs_cls.svg', dpi=400)
plt.close(fig)

fig = plot_barhist(result_df, 'mw')
fig.gca().set_xlabel('weight (Da)'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}mw_vs_cls.svg', dpi=400)
plt.close(fig)

fig = plot_barhist(result_df, 'nb_fragments')
fig.gca().set_xlabel('# fragments'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}nfrag_vs_cls.svg', dpi=400)
plt.close(fig)

# ---plot top 95 percent ---
fig = plot_barhist(result_df, 'seq_len', top_pct=0.95)
fig.gca().set_xlabel('sequence length'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}seqlen_vs_cls_top95.svg', dpi=400)
plt.close(fig)

fig = plot_barhist(result_df, 'mw', top_pct=0.95)
fig.gca().set_xlabel('weight (Da)'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}mw_vs_cls_top95.svg', dpi=400)
plt.close(fig)

fig = plot_barhist(result_df, 'nb_fragments', top_pct=0.95)
fig.gca().set_xlabel('# fragments'); fig.gca().set_ylabel('fraction')
plt.savefig(f'{out_dir}nfrag_vs_cls_top95.svg', dpi=400)
plt.close(fig)

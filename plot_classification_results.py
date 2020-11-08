import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from helpers import parse_output_dir

lab_order = ['top1', 'top3', 'misclassified']

def plot_barhist(cdf, feat, top_pct=None, binwidth=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    if top_pct:
        cdf = cdf.sort_values([feat]).head(int(len(cdf) * top_pct))
    sns.histplot(cdf, x=feat, hue='pred', stat='probability', multiple='fill',
                 hue_order=lab_order, binwidth=binwidth, element='step', linewidth=0, palette=['red', 'yellow', 'blue'],
                 ax=ax)
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
    if classify_dict[pid][0] == pid: pred = 'top1'
    elif pid in classify_dict[pid][:3]: pred = 'top3'
    else: pred = 'misclassified'
    result_df.loc[pid, :] = target_df.loc[pid].to_list() + [pred]
result_df.to_csv(f'{out_dir}classification_eval.csv')

# --- plot entire set ---
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

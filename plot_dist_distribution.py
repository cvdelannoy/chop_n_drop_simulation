import argparse, os, sys, pickle
import numpy as np
from os.path import splitext
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

palette = {False: '#fbb4ae', True: '#ccebc5'}

def load_target_stats_df(pkl_fn):
    with open(pkl_fn, 'rb') as fh: target_dict = pickle.load(fh)
    for nfrag in target_dict:
        target_dict[nfrag].drop([f'ss{i}' for i in range(nfrag)], inplace=True, axis=1)
        target_dict[nfrag].loc[:, 'nb_fragments'] = nfrag
    target_df = pd.concat(target_dict.values())
    return target_df

parser = argparse.ArgumentParser(description='Plot alignment distances for first-hit fingerprints')
parser.add_argument('--fp-pkl', type=str, required=True)
parser.add_argument('--targets-pkl', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)
args = parser.parse_args()

with open(args.fp_pkl, 'rb') as fh: fp_dict = pickle.load(fh)
target_df = load_target_stats_df(args.targets_pkl)

fp_dict = {fp: fp_dict[fp] for fp in fp_dict if len(fp_dict[fp])}
plot_df = pd.DataFrame(index=list(fp_dict), columns=['correct','dist'])
for fp in fp_dict:
    if fp not in target_df.index: continue
    # if not len(fp_dict[fp]): continue
    plot_df.loc[fp, :] = [fp_dict[fp][0][0] == fp, fp_dict[fp][0][1] / target_df.loc[fp, 'seq_len']]
plot_df.sort_values('dist', inplace=True, ascending=True)
plot_df.loc[:, 'idx'] = np.arange(len(plot_df))
plot_df.to_csv(splitext(args.out_svg)[0] + '.csv')

meta_agg_list = []
for corr, cdf in plot_df.groupby('correct'):
    nbp_per_point = len(cdf) / 100
    agg_list = []
    for ci, ccdf in cdf.reset_index().groupby(np.arange(len(cdf)) // nbp_per_point):
        agg_list.append(pd.Series({'dist': ccdf.dist.mean(), 'correct': corr}))
    adf = pd.concat(agg_list, axis=1).T
    adf.sort_values('dist', inplace=True)
    adf.loc[:, 'idx'] = np.arange(len(adf))
    meta_agg_list.append(adf)
agg_df = pd.concat(meta_agg_list)

agg_false_df, agg_true_df = agg_df.query('correct == False'), agg_df.query('correct')
dist_list = [np.mean([agg_false_df.loc[idx,'dist'], agg_true_df.loc[idx,'dist']])  for idx in np.arange(100)]

acc_df = pd.DataFrame(index=dist_list, columns=['acc'])
acc_df.index.rename('dist', inplace=True)
for dist in dist_list:
    cdf = plot_df.query('dist < @dist')
    acc_df.loc[dist, 'acc'] = len(cdf.query('correct')) / len(cdf)
acc_df.reset_index(inplace=True)
acc_df.acc = acc_df.acc.astype(float)

max_acc_dist = acc_df.loc[acc_df.acc.idxmax(), 'dist']

# --- summary lollipop graph ---
wdim = 4
fig = plt.figure(figsize=(8.25, 2.9375))
gs = gridspec.GridSpec(1, wdim, figure=fig)
ax = fig.add_subplot(gs[0, :wdim - 1])
cols = agg_df.correct.apply(lambda x: palette[x]).to_list()
plt.vlines(x=agg_df.loc[:,'idx'], ymin=0, ymax=agg_df.loc[:,'dist'], color=cols, lw=1)
plt.scatter(x=agg_df.loc[:,'idx'], y=agg_df.loc[:,'dist'], color=cols, s=2)
plt.axhline(max_acc_dist, lw=0.8, ls='dashed', color='black')

plt.xlim(0, 100)
plt.ylim(0, agg_df.dist.max() + 0.5)

plt.ylabel('S / residue')
plt.xlabel('Proteins (%)')

# --- accuracy vs cutoff graph ---

acc_df.to_csv(splitext(args.out_svg)[0] + '_acc.csv')

ax = fig.add_subplot(gs[0, wdim - 1:], sharey=ax)
sns.lineplot(x='acc', y='dist',
             sort=False, lw=1, markers=['o'], color='black',
             data=acc_df, ax=ax)
plt.xticks([0.65, 0.70, 0.75, 0.80])
plt.xlim(0.6, acc_df.acc.max()+0.01)
plt.axhline(max_acc_dist, lw=0.8, ls='dashed', color='black')
plt.xlabel('Accuracy (%)')

# ax.set_xticks([],minor=[])
# ax.get_legend().remove()
plt.savefig(args.out_svg)

# # --- full chart ---
#
# fig, ax = plt.subplots(figsize=(8.25, 2.9375))
# # cols = plot_df.correct.apply(lambda x: palette[x]).to_list()
# # plt.vlines(x=plot_df.loc[:,'idx'], ymin=0, ymax=plot_df.loc[:,'dist'], color=cols, lw=1)
# # plt.scatter(x=plot_df.loc[:,'idx'], y=plot_df.loc[:,'dist'], color=cols, s=2)
#
# plt.axhline(plot_df.query('correct').dist.mean(), color=palette[True], ls='--', lw=0.75, zorder=0)
# plt.axhline(plot_df.query('correct == False').dist.mean(), color=palette[False], ls='--', lw=0.75, zorder=1)
#
# for pid, tup in plot_df.iterrows():
#     ax.add_artist(Rectangle((tup.idx, 0), width=1, height=tup.dist, color=palette[tup.correct], zorder=tup.idx+2))
#
# plt.xlim(0, len(plot_df))
# plt.ylim(0, plot_df.dist.max())
#
#
# plt.ylabel('S / residue')
# plt.xlabel('')
# ax.set_xticks([],minor=[])
# # ax.get_legend().remove()
# plt.savefig(args.out_svg)

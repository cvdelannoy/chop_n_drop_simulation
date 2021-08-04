import argparse
from os.path import dirname, splitext
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helpers import parse_output_dir


parser = argparse.ArgumentParser(description='plot all violin plots in a single figure')
parser.add_argument('--in-csv', nargs='+', type=str, required=True)
parser.add_argument('--param-names', nargs='+', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)
args = parser.parse_args()

assert len(args.in_csv) == len(args.param_names)

nb_plots = len(args.in_csv)
fig = plt.figure(figsize=(8.25, 2.9375))
gs = gridspec.GridSpec(1,nb_plots, figure=fig, wspace=0.025)

for it, (param_name, csv_fn) in enumerate(zip(args.param_names, args.in_csv)):
    if it == 0:
        ax = fig.add_subplot(gs[0, it])
        first_ax = ax
    else:
        ax = fig.add_subplot(gs[0, it], sharey=first_ax)
    cdf = pd.read_csv(csv_fn, names=['idx', 'param', 'acc'], skiprows=1)
    cdf.acc = cdf.acc * 100
    if 'Resolution' in param_name:
        cdf.param = cdf.param.astype(int)
    sns.violinplot(x='param', y='acc',
                   color='grey', linewidth=0.1,
                   inner=None, data=cdf)
    # sns.boxplot(x='param', y='acc', data=cdf)
    # sns.pointplot(x='param', y='acc', capsize=0.3,
    #               scale=0.4,
    #               errwidth=1,
    #               ci=None,
    #               join=True, data=cdf)
    sns.stripplot(x='param', y='acc', color='black', size=2, jitter=False, data=cdf)
    ax.set_xlabel(param_name)
    if it == 0:
        plt.ylabel('Accuracy (%)')
    else:
        # ax.set_yticks([], [])
        # ax.set_yticklabels([])
        ax.set_ylabel('')
# plt.tight_layout()
plt.savefig(args.out_svg)

import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import parse_output_dir

parser = argparse.ArgumentParser(description='Plot chop n drop simulation comparison results at several values of a'
                                             'given parameter.')
parser.add_argument('--in-csv', nargs='+', type=str)
parser.add_argument('--range', nargs='+', type=float, help='Range of values for x axis')
parser.add_argument('--param-name', type=str, required=True)
parser.add_argument('--out-dir', type=str)

args = parser.parse_args()

assert len(args.range) == len(args.in_csv)
out_dir = parse_output_dir(args.out_dir)
out_dir_top95 = parse_output_dir(out_dir+'top95')
out_dir_all = parse_output_dir(out_dir+'all')

compare_df_list = []
for it, fn in enumerate(args.in_csv):
    df = pd.read_csv(fn)
    df.loc[:, args.param_name] = args.range[it]
    compare_df_list.append(df)
compare_df = pd.concat(compare_df_list)

lab_order = ['misclassified', 'top3', 'top1']
summary_df = pd.DataFrame(0, columns=lab_order, index=args.range)
summary_df.index.name = args.param_name

for res, cdf in compare_df.groupby(args.param_name):
    summary_df.loc[res, :] = cdf.pred.value_counts() / len(cdf)
summary_df.fillna(0.0, inplace=True)
summary_df.top3 = summary_df.top3 + summary_df.top1
summary_df = summary_df.astype(float)


summary_df.to_csv(f'{out_dir}uniqueness_vs_{args.param_name}.csv', header=True, index=True)

# --- plot ---
summary_df.reset_index(inplace=True)
plot_df = summary_df.melt(value_vars=['top3', 'top1'],
                          id_vars=[args.param_name], var_name='pred_cls', value_name='fraction')
fig = sns.lineplot(x=args.param_name, y='fraction', hue='pred_cls', data=plot_df)
plt.savefig(f'{out_dir}accuracy_vs_{args.param_name}.svg', dpi=400)

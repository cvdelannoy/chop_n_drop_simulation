import argparse, re
from os.path import dirname, splitext
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import parse_output_dir

parser = argparse.ArgumentParser(description='Plot chop n drop simulation comparison results at several values of a'
                                             'given parameter.')
parser.add_argument('--in-csv', nargs='+', type=str)
parser.add_argument('--param-name', type=str, required=True)
parser.add_argument('--out-svg', type=str)

args = parser.parse_args()
out_dir = parse_output_dir(dirname(args.out_svg))

summary_list = []
for it, fn in enumerate(args.in_csv):
    df = pd.read_csv(fn)
    param_value = re.search(f'(?<={args.param_name})[0-9.]+', fn).group(0)
    summary_list.append(pd.Series({args.param_name: param_value, 'acc': len(df.query('pred == "correct"')) / len(df)}))
summary_df = pd.concat(summary_list, axis=1).T
summary_df.acc = summary_df.acc.astype(float)

# --- plot ---
plt.figure(figsize=[10, 5])
sns.violinplot(x=args.param_name, y='acc', inner='stick', color='white', data=summary_df)
plt.ylabel('Accuracy (%)')
if '_' in args.param_name:
    plt.xlabel(args.param_name.replace('_', ' '))

plt.savefig(args.out_svg, dpi=400)
summary_df.to_csv(f'{splitext(args.out_svg)[0]}.csv')

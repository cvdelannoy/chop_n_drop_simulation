import argparse
import numpy as np

try:
    import pickle5 as pickle
except:
    import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from helpers import parse_output_dir

parser = argparse.ArgumentParser(description='Plot target fingerprints and found matches as spike trains')
parser.add_argument('--in-pkl', type=str,required=True)
parser.add_argument('--dynamic-range', type=float, nargs=2, default=[500, 2000])
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
plot_dir = parse_output_dir(out_dir+ 'plots')
plot_correct_dir = parse_output_dir(out_dir + 'correctly_classified')
plot_incorrect_dir = parse_output_dir(out_dir + 'incorrectly_classified')

with open(args.in_pkl, 'rb') as fh: match_dict = pickle.load(fh)

diffs = []
nb_out_range = []
for target_id in match_dict:
    plt.figure(figsize=(10,5))
    cur_dict = match_dict[target_id]
    id_list = list(match_dict[target_id])
    frag_lens = [np.cumsum(m) for m in match_dict[target_id].values()]

    # If right fingerprint was found store:
    # - differences
    # - number of fragments outside dynamic range
    id_list_matches = id_list.copy()
    id_list_matches.remove('target')
    best_match = [idm.split('_')[0] for idm in id_list_matches if '_0' in idm][0]
    id_list_matches_dict = {idm.split('_')[0]: idm for idm in id_list_matches}
    if target_id in id_list_matches_dict:
        target_fp = np.array(match_dict[target_id]['target'])
        match_fp = np.array(match_dict[target_id][id_list_matches_dict[target_id]])
        match_fp_range_bool = np.logical_and(match_fp > args.dynamic_range[0], match_fp < args.dynamic_range[1])
        target_fp_range_bool = np.logical_and(target_fp > args.dynamic_range[0], target_fp < args.dynamic_range[1])
        nb_out_range.append(np.sum(np.invert(match_fp_range_bool)))
        if np.sum(target_fp_range_bool) == np.sum(match_fp_range_bool):
            diffs.append(target_fp[target_fp_range_bool] - match_fp[match_fp_range_bool])

    # Plot
    plt.eventplot(frag_lens)
    plt.yticks(list(range(len(id_list))), id_list)
    plt.xlabel('Weight (Da)')
    if target_id == best_match:
        plt.savefig(f'{plot_correct_dir}{target_id}.svg', dpi=400)
    else:
        plt.savefig(f'{plot_incorrect_dir}{target_id}.svg', dpi=400)
    plt.close(fig=plt.gcf())

diff_array = np.concatenate(diffs)
gmm = GaussianMixture(n_components=1).fit(diff_array[np.logical_and(diff_array > -200, diff_array < 200)].reshape(-1,1))
sd = np.sqrt(gmm.covariances_[0,0])
mean = gmm.means_[0][0]

plt.hist(diff_array,  bins=100)
plt.xlabel('Weight difference (Da)')
plt.xlim((-200,200))
plt.gca().text(-175, 500, f'mean={mean.round(2)}\nsd={sd.round(2)}', fontsize=14, verticalalignment='top')
plt.savefig(f'{out_dir}diff_hist.svg')
plt.close(fig=plt.gcf())

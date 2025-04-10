# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:00:43 2024

@author: NaN
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from kilosort.run_kilosort import load_sorting

# Path to load existing sorting results from.
results_dir = Path('Z:/kilosort/m24c112/')
# Paths where new labels will be saved for use with Phy.
save_1 = results_dir / 'cluster_KSLabel.tsv'
save_2 = results_dir / 'cluster_group.tsv'
# Save a backup of KS4's original labels before overwriting (recommended).
shutil.copyfile(save_1, results_dir / 'cluster_KSLabel_backup.tsv')

# Load sorting results
ops, st, clu, similar_templates, is_ref, est_contam_rate = load_sorting(results_dir)

cluster_labels = np.unique(clu)  # integer label for each cluster
fs = ops['fs']                   # sampling rate

# Option 1: Use existing labels as a starting point.
#           KS4 assigns "good" where is_ref is True, and "mua" otherwise.
# label_good = is_ref.copy()

# Option 2: Ignore KS4's labels and only use your own criteria.
label_good = np.zeros(cluster_labels.size)

# criteria 1 contamination rate
# contam_good = est_contam_rate < 0.2   # this already has shape (n_clusters,)
contam_good = np.ones(cluster_labels.size)

# criteria 2 firing rate
fr_good = np.zeros(cluster_labels.size, dtype=bool)
fr = np.zeros(cluster_labels.size)
for i, c in enumerate(cluster_labels):
    # Get all spikes assigned to this cluster
    spikes = st[clu == c]
    # Compute est. firing rate using your preferred method.
    # Note that this formula will not work well for units that drop in and out.
    fr[i] = spikes.size / (spikes.max()/fs - spikes.min()/fs)
    if fr[i] >= 2:
        fr_good[i] = True

# Update labels, requiring that all criteria hold for each cluster.
label_good = np.logical_and(contam_good, fr_good)

# criteria 3 units presence ratio
# Formula adapted from https://github.com/AllenInstitute/ecephys_spike_sorting/
def presence_ratio(spike_train, num_bins, min_time, max_time, min_spike_pct=0.05):
    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))
    min_spikes = h.mean()*min_spike_pct

    # NOTE: Allen Institute formula leaves off the -1 to force the ratio to
    #       never reach 1.0. We've included it here because without it the ratio
    #       is biased too much for a small number of bins.
    return np.sum(h > min_spikes) / (num_bins - 1)

# Compute presence ratio for each cluster
presence = np.zeros(cluster_labels.size)
min_time = st.min()
max_time = st.max()
for i, c in enumerate(cluster_labels):
    spikes = st[clu == c]
    presence[i] = presence_ratio(spikes, 10, min_time, max_time)

presence_good = presence >= 1
# Update labels with the additional criteria.
label_good = np.logical_and(label_good, presence_good)

# Convert True/False to 'good'/'mua'
ks_labels = ['good' if b else 'mua' for b in label_good]

good_index = [index for index,value in enumerate(ks_labels) if value=="good"]
select_good = pd.DataFrame({
    'cluster_id': good_index,
    'group':['good']*len(good_index)
    })

# Write to two .tsv files.
with open(save_1, 'w') as f:
    f.write(f'cluster_id\tKSLabel\n')
    for i, p in enumerate(ks_labels):
        f.write(f'{i}\t{p}\n')
# shutil.copyfile(save_1, save_2)
select_good.to_csv(save_2,sep='\t', index=False)
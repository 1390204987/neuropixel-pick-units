import numpy as np
import os
import pandas as pd
from pathlib import Path
from phy.apps.base import BaseController
from phylib.io.model import load_model
from phylib.stats.ccg import correlograms

# 确保路径正确

results_dir = Path('Z:/kilosort/m24c110/')
save_2 = results_dir / 'cluster_group.tsv'
model_path = results_dir / 'params.py'

# 加载 model
model = load_model(model_path)
print("Model loaded. Sample rate:", model.sample_rate)

# get all units ID marked as good by kilosort picked
os.chdir(results_dir)
cluster_info = pd.read_csv('./cluster_group.tsv',sep='\t')
cluster_Amplitude = pd.read_csv('./cluster_Amplitude.tsv',sep='\t')
select_goodid = cluster_info["cluster_id"]
correlogram_data = {}
cluster_info_new = cluster_info.copy()
cluster_info_new.loc[cluster_info_new['group'] == 'good', 'group'] = 'mua'
#  do autocorrelogram for each markerd good id
for iunit in select_goodid:
    curr_id = iunit
    # curr_id = [328]
    # 选取符合 cluster 的 spike
    cluster_mask = np.isin(model.spike_clusters, curr_id)
    selected_spikes = np.where(cluster_mask)[0]
    
    # 限制最大 spike 数量，防止计算负担过大
    n_spikes = min(10000, len(selected_spikes))
    random_indices = np.random.choice(selected_spikes, n_spikes, replace=False)
    
    
    # 重新排序，确保 spike_times 递增
    random_indices.sort()
    
    # 获取 spike_times 和 spike_clusters
    spike_times = model.spike_times[random_indices]
    spike_clusters = model.spike_clusters[random_indices]
    
    # 选择 bin 和窗口大小
    bin_size = 0.001  # 1ms bins
    window_size = 0.05  # ±500ms 窗口
    
    # 计算 correlogram
    data = correlograms(spike_times, spike_clusters, 
                        cluster_ids=curr_id,sample_rate=model.sample_rate,bin_size=bin_size, 
                        window_size=window_size)
    correlogram_data[iunit] = np.squeeze(data)
    middle_index = int((window_size/bin_size)/2)
    num_fir_together = correlogram_data[iunit][middle_index]
    
    # get cluster amplitude
    curr_amplitude = cluster_Amplitude.loc[cluster_Amplitude['cluster_id']==curr_id,'Amplitude'].values[0]
    if num_fir_together <3 and curr_amplitude>=10:
        cluster_info_new.loc[cluster_info_new['cluster_id'] == curr_id, 'group'] = 'good'
        
cluster_info_new.to_csv(save_2,sep='\t', index=False)
# import matplotlib.pyplot as plt


# # 计算 bin_edges，使中央 bin 为 [0, 0.001]
# bin_edges = np.arange(-window_size, window_size + bin_size, bin_size)
# # 修正 bin_edges 以匹配 data 的长度
# if len(bin_edges) - 1 != len(correlogram_data):
#     bin_edges = bin_edges[:len(correlogram_data) + 1]
# # 绘制直方图
# plt.figure(figsize=(8, 6))
# plt.bar(bin_edges[:-1], correlogram_data, width=bin_size, color='blue', alpha=0.7)
# plt.title("Spike Time Autocorrelogram")
# plt.xlabel("Time Lag (s)")
# plt.ylabel("Spike Count")
# plt.grid(True)
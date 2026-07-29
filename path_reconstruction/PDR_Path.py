import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Data_processing
import Ore_smooth

matplotlib.use('TkAgg')


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BEST_K = 0.38
# 0.47




def preprocessing(data):
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S:%f', errors='coerce')
    data['Time_sec'] = (data['Time'] - data['Time'].min()).dt.total_seconds()
    
    data['Acc_Z_corrected'] = data['Acc_Z'] - 9.8
    
    data['Acc_mag'] = np.sqrt(data['Acc_X'] ** 2 + data['Acc_Y'] ** 2 + data['Acc_Z_corrected'] ** 2)
    data['Acc_mag'] -= data['Acc_mag'].mean()
    return data



def detect_amplitudes(acc_data, fs=50, peak_threshold_factor=1.0, window_size=10):
    mean_val = np.mean(acc_data)
    std_val = np.std(acc_data)
    peak_threshold = mean_val + peak_threshold_factor * std_val
    min_distance = int(0.4 * fs)  

    peaks, _ = find_peaks(acc_data, height=peak_threshold, distance=min_distance)

    amplitudes = []
    for peak in peaks:
        start = max(peak - window_size, 0)
        end = min(peak + window_size, len(acc_data))
        peak_val = acc_data[peak]
        valley_before = np.min(acc_data[start:peak]) if peak > 0 else peak_val
        valley_after  = np.min(acc_data[peak:end])   if peak < len(acc_data)-1 else peak_val
        A_i = peak_val - np.mean([valley_before, valley_after])
        amplitudes.append(A_i)
    return amplitudes, peaks



def reconstruct_path(step_lengths, step_angles):
    positions = [(0, 0)]
    for L, angle in zip(step_lengths, step_angles):
        x_prev, y_prev = positions[-1]
        dx = L * np.cos(np.deg2rad(angle))
        dy = L * np.sin(np.deg2rad(angle))
        positions.append((x_prev + dx, y_prev + dy))
    return np.array(positions)


def find_anchor(data, peaks):
    
    data[['Start', 'End']] = data['Anchor_Info'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
    
    start_times = data['Start'].to_numpy()
    closest_peaks = np.array([peaks[np.abs(peaks - t).argmin()] for t in start_times])
    peak_indices = np.array([np.where(peaks == p)[0][0] for p in closest_peaks])
    
    data['Start'] = peak_indices
    
    median_start_colum = data.groupby('Cluster_Label')['Start'].median().round().astype(int)

    
    Z = linkage(median_start_colum.values.reshape(-1, 1), method='complete')  
    threshold = 10  
    cluster_assignments = fcluster(Z, threshold, criterion='distance')  

    
    data['New_Cluster_Label'] = data['Cluster_Label'].map(dict(zip(median_start_colum.index, cluster_assignments)))

    
    grouped_medians = data.groupby('New_Cluster_Label')['Start'].median().round().astype(int)
    sorted_labels = {old: new for new, old in
                     enumerate(sorted(grouped_medians.index, key=lambda x: grouped_medians[x]))}

    data['Sorted_Cluster_Label'] = data['New_Cluster_Label'].map(sorted_labels)
    median_start_colum = grouped_medians.sort_values().to_numpy()

    
    # median_start_colum[1] += 3
    return median_start_colum



def get_turn(data, peaks):
    ore_raw = data['Ore'].copy().reset_index(drop=True)

    
    window_length = 25 if 25 % 2 == 1 else 26  
    ore_smoothed = pd.Series(savgol_filter(ore_raw, window_length=window_length, polyorder=2))

    
    angle_diffs = ore_smoothed.diff().fillna(0).to_numpy()

    
    angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
    angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)

    
    jump_threshold = 180  
    angle_diffs = np.where(np.abs(angle_diffs) > jump_threshold, 0, angle_diffs)

    
    window_size = 200  
    cumulative_diffs = pd.Series(angle_diffs).rolling(window=window_size, min_periods=1).sum().to_numpy()

    
    turn_threshold = 60  
    min_interval = 400  

    
    turn_info = []  
    prev_idx = -min_interval  

    for idx, cum_diff in enumerate(cumulative_diffs):
        if cum_diff > turn_threshold or cum_diff < -turn_threshold:
            if idx - prev_idx >= min_interval:
                
                direction = -1 if cum_diff > turn_threshold else 1

                
                peak_idx = np.argmin(np.abs(peaks - idx))

                
                turn_info.append([direction, peak_idx])
                prev_idx = idx

    return turn_info



def correction_turn(turn_info):
    for turn in turn_info:
        if turn[1] == 101:
              turn[0] = -1
    #     if turn[1] == 144:
    #           turn[0] = -1
    #     if turn[1] == 108:
    #         turn[0] = 1
    #     if turn[1] == 314:
    #           turn[0] = -1
    #     if turn[1] == 360:
    #         turn[0] = -1
    #     if turn[1] == 705:
    #         turn[0] = -1
    remove_rows = {55,168,194,207,270}
    turn_info = [turn for turn in turn_info if turn[1] not in remove_rows]
    return turn_info



def find_path_between_anchors(median_start_colum, turn_info, step_lengths):
    paths = []

    
    for i in range(len(median_start_colum) - 1):
        A = median_start_colum[i]
        B = median_start_colum[i + 1]

        
        path = [(i, i+1)]
        last_position = A  
        if_turn = 0  

        for turn in turn_info:
            if A < turn[1] < B:  
                
                distance = np.sum(step_lengths[last_position:turn[1]])  
                path.append((if_turn, int(round(distance))))  
                if_turn = turn[0]  
                last_position = turn[1]  

        
        if last_position < B:
            distance = np.sum(step_lengths[last_position:B])  
            path.append((if_turn, int(round(distance))))  

        paths.append(path)

    return paths



def show(path_ore, median_start_colum, turn_info, df):
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(path_ore[:, 0], path_ore[:, 1], '--', label='Ore-only reconstructed path', color='blue')
    # Anchor
    anchor_positions = path_ore[median_start_colum]  
    plt.plot(anchor_positions[:, 0], anchor_positions[:, 1], 'ro', markersize=8, label='Anchor')
    # Turn point
    turn_positions = [path_ore[turn[1], :] for turn in turn_info]
    turn_positions = np.array(turn_positions)  
    plt.plot(turn_positions[:, 0], turn_positions[:, 1], 'o', color='orange', markersize=8, label='Turn point')
    # Start
    start_position = path_ore[0, :]  
    plt.scatter(start_position[0], start_position[1], color='green', s=100, label='Start', zorder=5)

    
    plt.xlabel('X coordinate (m)', fontsize=16)
    plt.ylabel('Y coordinate (m)', fontsize=16)
    plt.title('Reconstructed path with anchors and turn points', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_sec'], df['Ore'], label='Raw Ore heading', color='green', linestyle='--')
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Heading (degrees)', fontsize=16)
    plt.title('Heading over time', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    # plt.show()



def PDR(df,Anchor_data,If_show=True):
    
    df = preprocessing(df)

    
    amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  

    
    step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
    total_distance = np.sum(step_lengths)  
    print(f"Estimated walking distance: {total_distance:.3f} m")

    
    # df['Ore_smooth'] = Ore_smooth.smooth_ore(df, ore_col='Ore', fs=50)
    ore_angles = df['Ore'].values
    step_angles_ore = ore_angles[peaks]
    
    angles_for_clustering = step_angles_ore.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=8, random_state=0).fit(angles_for_clustering)
    
    step_angles_ore = kmeans.cluster_centers_[kmeans.labels_].flatten()

    
    path_ore = reconstruct_path(step_lengths, step_angles_ore)

    
    median_start_colum = find_anchor(Anchor_data, peaks)

    turn_info = get_turn(df, peaks)
    turn_info = correction_turn(turn_info)
    print(median_start_colum)
    print(turn_info)
    paths = find_path_between_anchors(median_start_colum, turn_info, step_lengths)
    print(paths)


    if If_show==True:
        
        show(path_ore, median_start_colum, turn_info, df)

    return median_start_colum,turn_info,paths


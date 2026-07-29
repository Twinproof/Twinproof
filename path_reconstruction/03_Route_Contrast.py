import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from sklearn.decomposition import PCA
import Data_processing
from pathlib import Path

matplotlib.use('TkAgg')


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BEST_K2 = 0.4706652832031252
BEST_K=0.47



PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "data" / "collectionData" / "route_003" / "sensor_5.csv"
df = pd.read_csv(file_path)
Anchor_data = pd.read_csv(
    PROJECT_ROOT / "Find_Anchor" / "anchor_cluster" / "anchor_cluster_route_003_refined.csv"
)

file_path_2 = PROJECT_ROOT / "data" / "collectionData" / "route_004" / "sensor_5.csv"
df_2 = pd.read_csv(file_path_2)
Anchor_data_2 = pd.read_csv(
    PROJECT_ROOT / "Find_Anchor" / "anchor_cluster" / "anchor_cluster_route_004_refined.csv"
)


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

    
    # median_start_colum[3] -= 10
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

    
    turn_threshold = 50  
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
        if turn[1] == 471:
              turn[0] = 1
        if turn[1] == 360:
            turn[0] = -1
        if turn[1] == 705:
            turn[0] = -1
    remove_rows = {597,607 , 628}
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



def align_paths_by_start_and_direction(path_src, path_target, steps_for_direction=10):
    """Process the inputs and return the corresponding result."""
    
    delta = path_target[0] - path_src[0]
    path_src_aligned = path_src + delta

    
    if len(path_src_aligned) < steps_for_direction or len(path_target) < steps_for_direction:
        return path_src_aligned  

    vec_src = path_src_aligned[steps_for_direction] - path_src_aligned[0]
    vec_tgt = path_target[steps_for_direction] - path_target[0]

    
    angle_src = np.arctan2(vec_src[1], vec_src[0])
    angle_tgt = np.arctan2(vec_tgt[1], vec_tgt[0])
    angle_diff = angle_tgt - angle_src

    
    rotation_matrix = np.array([
        [np.cos(angle_diff), -np.sin(angle_diff)],
        [np.sin(angle_diff),  np.cos(angle_diff)]
    ])
    rotated = (rotation_matrix @ (path_src_aligned - path_src_aligned[0]).T).T + path_src_aligned[0]

    return rotated




def show_two_paths(path_ore_1, anchors_1, turns_1, df_1,
                   path_ore_2, anchors_2, turns_2, df_2):
    plt.figure(figsize=(12, 10))

    
    plt.plot(path_ore_1[:, 0], path_ore_1[:, 1], '--', label='Path 1 (device 1)', color='blue')
    anchor_pos_1 = path_ore_1[anchors_1]
    plt.plot(anchor_pos_1[:, 0], anchor_pos_1[:, 1], 'ro', label='Anchor 1', markersize=8)

    turn_pos_1 = np.array([path_ore_1[t[1]] for t in turns_1])
    # if len(turn_pos_1) > 0:
    #     plt.plot(turn_pos_1[:, 0], turn_pos_1[:, 1], 'o', color='orange', label='Turn point1', markersize=8)

    
    plt.plot(path_ore_2[:, 0], path_ore_2[:, 1], '--', label='Path 2 (device 2)', color='green')
    anchor_pos_2 = path_ore_2[anchors_2]
    plt.plot(anchor_pos_2[:, 0], anchor_pos_2[:, 1], 'mo', label='Anchor 2', markersize=8)

    turn_pos_2 = np.array([path_ore_2[t[1]] for t in turns_2])
    # if len(turn_pos_2) > 0:
    #     plt.plot(turn_pos_2[:, 0], turn_pos_2[:, 1], 'o', color='orange', label='Turn point2', markersize=8)

    
    start = path_ore_1[0]
    plt.scatter(start[0], start[1], color='black', s=100, label='Start', zorder=5)

    
    plt.xlabel('X coordinate (m)', fontsize=16)
    plt.ylabel('Y coordinate (m)', fontsize=16)
    plt.title('Path comparison with anchors and turn points', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(14, 6))
    plt.plot(df_1['Time_sec'], df_1['Ore'], label='Device 1 heading', linestyle='--', color='blue')
    plt.plot(df_2['Time_sec'], df_2['Ore'], label='Device 2 heading', linestyle='--', color='green')
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Heading (degrees)', fontsize=16)
    plt.title('Heading comparison', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()




df = preprocessing(df)
df_2 = preprocessing(df_2)


amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  
amplitudes_2, peaks_2 = detect_amplitudes(df_2['Acc_mag'].values, fs=50)  


step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
total_distance = np.sum(step_lengths)  

step_lengths_2 = [BEST_K2 * (A_i ** 0.25) for A_i in amplitudes_2 if A_i > 0]
total_distance_2 = np.sum(step_lengths_2)  


ore_angles = df['Ore'].values
step_angles_ore = ore_angles[peaks]  

ore_angles_2 = df_2['Ore'].values
step_angles_ore_2 = ore_angles_2[peaks_2]  



path_ore = reconstruct_path(step_lengths, step_angles_ore)
path_ore_2 = reconstruct_path(step_lengths_2, step_angles_ore_2)

path_ore_2 = align_paths_by_start_and_direction(path_ore_2, path_ore, steps_for_direction=10)



median_start_colum = find_anchor(Anchor_data, peaks)
turn_info = get_turn(df, peaks)
paths = find_path_between_anchors(median_start_colum, turn_info, step_lengths)

median_start_colum_2 = find_anchor(Anchor_data_2, peaks_2)
turn_info_2 = get_turn(df_2, peaks_2)
paths_2 = find_path_between_anchors(median_start_colum_2, turn_info_2, step_lengths_2)



show_two_paths(
    path_ore, median_start_colum, turn_info, df,
    path_ore_2, median_start_colum_2, turn_info_2, df_2
)

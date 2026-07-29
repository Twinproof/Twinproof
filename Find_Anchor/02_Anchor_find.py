import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict
import numpy as np
import os
import matplotlib
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
folder_path = PROJECT_ROOT / "data" / "collectionData_02" / "route_004"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
output_file_path = SCRIPT_DIR / "anchor" / "anchor_combined_route_004.csv"
output_folder = os.path.dirname(output_file_path)
if output_folder:
    os.makedirs(output_folder, exist_ok=True)

window_size = 400
step_size = 10

all_segment_data = []
anchor_points_info = []

def time_to_seconds(time_values):
    """Convert mmm values to seconds since midnight."""
    parts = time_values.astype(str).str.extract(
        r'^(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}):(?P<millisecond>\d{3})$'
    )
    parts = parts.apply(pd.to_numeric, errors='coerce')
    valid = (
        parts['hour'].between(0, 23)
        & parts['minute'].between(0, 59)
        & parts['second'].between(0, 59)
        & parts['millisecond'].between(0, 999)
    )
    seconds = (
        (parts['hour'] * 60 + parts['minute']) * 60
        + parts['second']
        + parts['millisecond'] / 1000
    )
    return seconds.where(valid)

def find_anchor(data, window_size=400, step_size=10, mode='or'):
    data['Time'] = time_to_seconds(data['Time'])
    data.dropna(subset=['Time'], inplace=True)
    data['Time'] = data['Time'] - data['Time'].min()

    # Smooth all five RSSI channels.
    for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3', 'Cell_RSSI_4', 'Cell_RSSI_5']:
        data[col] = data[col].ffill().bfill()
        data[col] = data[col].rolling(window=5, min_periods=1).mean()

    # Fill missing values in all five cell ID channels.
    for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3', 'Cell_ID_4', 'Cell_ID_5']:
        data[col] = data[col].ffill().bfill()

    data['Meg_diff1'] = data['Meg'].diff().rolling(window=5, min_periods=1).mean()
    scaler = StandardScaler()
    data['Meg_diff1_scaler'] = scaler.fit_transform(data[['Meg_diff1']])
    data['Meg_diff1_abs'] = np.abs(data['Meg_diff1_scaler'])

    mean_value = data['Meg_diff1_abs'].mean()
    std_value = data['Meg_diff1_abs'].std()
    upper_bound_meg = mean_value + 0.07 * std_value

    peaks, _ = find_peaks(data['Meg_diff1_scaler'], distance=50)
    valleys, _ = find_peaks(-data['Meg_diff1_scaler'], distance=50)
    upper_bound_peaks = data['Meg_diff1_scaler'].iloc[peaks].mean() + 0.6 * data['Meg_diff1_scaler'].iloc[peaks].std()
    under_bound_valley = data['Meg_diff1_scaler'].iloc[valleys].mean() - data['Meg_diff1_scaler'].iloc[valleys].std()

    anchor_points = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end]

        mean_meg_diff1 = window['Meg_diff1_abs'].mean()
        peak_values = window['Meg_diff1_scaler'].iloc[find_peaks(window['Meg_diff1_scaler'], distance=50)[0]]
        valley_values = window['Meg_diff1_scaler'].iloc[find_peaks(-window['Meg_diff1_scaler'], distance=50)[0]]
        max_peak = peak_values.max() if len(peak_values) > 0 else 0
        min_valley = valley_values.min() if len(valley_values) > 0 else 0
        rolling_std = window['Meg_diff1_scaler'].rolling(window=20, min_periods=1).std().mean()
        # skewness_val = skew(window['Meg_diff1_scaler'].dropna())

        # Average RSSI variability across five channels.
        lte_rssi_std = np.mean([
            window['Cell_RSSI_1'].std(),
            window['Cell_RSSI_2'].std(),
            window['Cell_RSSI_3'].std(),
            window['Cell_RSSI_4'].std(),
            window['Cell_RSSI_5'].std()
        ])

        # Compute the cell ID change rate across five channels.
        id_changes = 0
        for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3', 'Cell_ID_4', 'Cell_ID_5']:
            ids = window[col].fillna(-1).astype(int).values
            id_changes += np.count_nonzero(np.diff(ids) != 0)
        lte_id_change_rate = id_changes / (5 * (window_size - 1))

        meg_flag = (mean_meg_diff1 >= upper_bound_meg and max_peak >= upper_bound_peaks
                    and min_valley <= under_bound_valley and rolling_std >= 0.08)
        lte_flag = (lte_rssi_std > 2.0 or lte_id_change_rate > 0.02)

        if (mode == 'or' and (meg_flag or lte_flag)) or (mode == 'and' and (meg_flag and lte_flag)):
                anchor_points.append((start, end))

    merged_segments = []
    for seg in sorted(anchor_points):
        if not merged_segments or merged_segments[-1][1] < seg[0]:
            merged_segments.append(seg)
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], seg[1]))

    final_segments = []
    for seg in merged_segments:
        if seg[1] - seg[0] > window_size:
            max_index = data['Meg_diff1_abs'].iloc[seg[0]:seg[1]].idxmax()
            new_start = max(max_index - window_size // 2, seg[0])
            new_end = min(new_start + window_size, seg[1])
            final_segments.append((new_start, new_end))
        else:
            final_segments.append(seg)

    filtered_segments = []
    i = 0
    while i < len(final_segments) - 1:
        if final_segments[i + 1][0] - final_segments[i][1] <= 600:
            combined_start = final_segments[i][0]
            combined_end = final_segments[i + 1][1]
            max_index = data['Meg_diff1_abs'].iloc[combined_start:combined_end].idxmax()
            new_start = max(max_index - window_size // 2, combined_start)
            new_end = min(new_start + window_size, combined_end)
            filtered_segments.append((new_start, new_end))
            i += 2
        else:
            filtered_segments.append(final_segments[i])
            i += 1
    if i == len(final_segments) - 1:
        filtered_segments.append(final_segments[i])

    return filtered_segments

def plot_signal_with_anchor_points(data, filtered_segments, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['Meg'], label='Meg Signal', color='blue')
    for start, end in filtered_segments:
        plt.axvspan(data['Time'].iloc[start], data['Time'].iloc[end], color='red', alpha=0.5)
    plt.title(f'Meg Signal and Anchor Points - {file_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Meg Signal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_lte_with_anchor_points(data, filtered_segments, file_name):
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    lte_cols = ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3', 'Cell_RSSI_4', 'Cell_RSSI_5']

    for idx, col in enumerate(lte_cols):
        if col in data.columns:
            plt.plot(data['Time'], data[col], label=col, color=colors[idx], alpha=0.7)

    for start, end in filtered_segments:
        plt.axvspan(data['Time'].iloc[start], data['Time'].iloc[end], color='yellow', alpha=0.3, label='Anchor Region')

    plt.title(f'LTE RSSI and Anchor Points - {file_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('RSSI (dBm)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    data_copy = data.copy(deep=True)
    filtered_segments = find_anchor(data_copy)

    data['Time'] = time_to_seconds(data['Time'])
    data.dropna(subset=['Time'], inplace=True)
    data['Time'] = data['Time'] - data['Time'].min()
    # plot_signal_with_anchor_points(data, filtered_segments, file_name)
    print(f"{file_name} anchor segments: {[(s, e) for s, e in filtered_segments]}")

    for seg_start, seg_end in filtered_segments:
        segment = data['Meg'].iloc[seg_start:seg_end].values[:400]
        if len(segment) < 400:
            segment = list(segment) + [None] * (400 - len(segment))

        pre_anchor_segment = data['Meg'].iloc[max(0, seg_start - 400):seg_start].values[-400:]
        if len(pre_anchor_segment) < 400:
            pre_anchor_segment = [None] * (400 - len(pre_anchor_segment)) + list(pre_anchor_segment)

        post_anchor_segment = data['Meg'].iloc[seg_end:seg_end + 400].values[:400]
        if len(post_anchor_segment) < 400:
            post_anchor_segment = list(post_anchor_segment) + [None] * (400 - len(post_anchor_segment))

        # Save segments from all five RSSI channels.
        lte_segments = []
        for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3', 'Cell_RSSI_4', 'Cell_RSSI_5']:
            lte_segment = data[col].iloc[max(0, seg_start - 4):min(len(data), seg_end + 4)].values[:12]
            if len(lte_segment) < 12:
                lte_segment = list(lte_segment) + [None] * (12 - len(lte_segment))
            lte_segments.extend(lte_segment)

        # Save segments from all five cell ID channels.
        lte_id_segments = []
        for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3', 'Cell_ID_4', 'Cell_ID_5']:
            id_segment = data[col].iloc[max(0, seg_start - 4):min(len(data), seg_end + 4)].values[:12]
            if len(id_segment) < 12:
                id_segment = list(id_segment) + [None] * (12 - len(id_segment))
            lte_id_segments.extend(id_segment)

        mid_time = (seg_start + seg_end) // 2
        mid_time_str = pd.to_datetime(data['Time'].iloc[mid_time], unit='s').strftime('%H:%M:%S')

        combined_segment = (
            list(segment)
            + list(pre_anchor_segment)
            + list(post_anchor_segment)
            + lte_segments
            + lte_id_segments
        )
        combined_segment.append(file_name)
        combined_segment.append(mid_time_str)
        all_segment_data.append(combined_segment)
        anchor_points_info.append((file_name, seg_start, seg_end))

columns = (
    [f'Column_{i + 1}' for i in range(400)]
    + [f'Pre_Anchor_{i + 1}' for i in range(400)]
    + [f'Post_Anchor_{i + 1}' for i in range(400)]
    + [f'Cell_RSSI_1_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_2_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_3_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_4_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_5_{i + 1}' for i in range(12)]
    + [f'Cell_ID_1_{i + 1}' for i in range(12)]
    + [f'Cell_ID_2_{i + 1}' for i in range(12)]
    + [f'Cell_ID_3_{i + 1}' for i in range(12)]
    + [f'Cell_ID_4_{i + 1}' for i in range(12)]
    + [f'Cell_ID_5_{i + 1}' for i in range(12)]
    + ['File_Name']
    + ['Mid_Time']
)

segment_df = pd.DataFrame(all_segment_data, columns=columns)
anchor_info_df = pd.DataFrame(anchor_points_info, columns=['File', 'Start_Index', 'End_Index'])
segment_df['Anchor_Info'] = anchor_info_df.apply(lambda x: f"({x['Start_Index']},{x['End_Index']})", axis=1)

segment_df.to_csv(output_file_path, index=False)
print(f"\nAnchor segments saved to: {output_file_path}")

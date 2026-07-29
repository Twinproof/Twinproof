import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from pathlib import Path

# Configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
folder_path = PROJECT_ROOT / "data" / "collectionData_02" / "route_004"
labels = ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']
TARGET_LEN = 200  # Target sample length used to limit processing time.

csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort()

def extract_rssi_columns(file_path):
    """Extract RSSI columns and resample them to TARGET_LEN values."""
    df = pd.read_csv(file_path)
    rssi_dict = {}
    for col in labels:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy()
            if len(vals) > 0:
                x = np.linspace(0, 1, len(vals))
                x_new = np.linspace(0, 1, TARGET_LEN)
                vals = np.interp(x_new, x, vals)
            rssi_dict[col] = vals
        else:
            rssi_dict[col] = np.array([])
    return rssi_dict

# Load and resample the input data.
rssi_data = {f: extract_rssi_columns(f) for f in csv_files if any(len(v) > 0 for v in extract_rssi_columns(f).values())}
file_list = list(rssi_data.keys())
print(f"Loaded {len(file_list)} valid files for DTW comparison ({TARGET_LEN} samples each)")

def avg_dtw_distance(file1, file2):
    """Calculate and average DTW distances for the three RSSI columns."""
    dists = []
    for col in labels:
        seq1 = rssi_data[file1][col]
        seq2 = rssi_data[file2][col]
        if len(seq1) > 0 and len(seq2) > 0:
            dist, _ = fastdtw(seq1, seq2, dist=lambda a, b: abs(a - b))
            dists.append(dist)
    return np.mean(dists) if dists else np.nan

# Calculate pairwise average distances.
distances = []
for i in range(len(file_list)):
    for j in range(i + 1, len(file_list)):
        avg_dist = avg_dtw_distance(file_list[i], file_list[j])
        if not np.isnan(avg_dist):
            distances.append((file_list[i], file_list[j], avg_dist))

# Calculate the mean DTW distance for each file.
avg_distances = {f: [] for f in file_list}
for f1, f2, d in distances:
    avg_distances[f1].append(d)
    avg_distances[f2].append(d)

avg_distances = {f: np.mean(dlist) for f, dlist in avg_distances.items()}
mean_dist = np.mean(list(avg_distances.values()))
std_dist = np.std(list(avg_distances.values()))
threshold = mean_dist + std_dist

print(f"\nMean DTW distance: {mean_dist:.2f}, threshold: {threshold:.2f}")
print("Outlier files:")
for f, avg_d in avg_distances.items():
    if avg_d > threshold:
        print(f" - {os.path.basename(f)} (mean DTW: {avg_d:.2f})")

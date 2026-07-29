import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from pathlib import Path
from scipy.spatial.distance import euclidean

# Directory containing the preprocessed input files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
folder_path = PROJECT_ROOT / "data" / "collectionData_02" / "route_004"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

meg_series_list = []
file_names = []

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path, usecols=['Meg'])
        # Convert the values to a one-dimensional floating-point array.
        meg_series = pd.to_numeric(df['Meg'], errors='coerce').dropna().to_numpy(dtype=float).ravel()

        if len(meg_series) == 0:
            print(f"[skip] {file_name} has an empty or invalid Meg column")
            continue

        meg_series_list.append(meg_series)
        file_names.append(file_name)
    except Exception as e:
        print(f"[error] failed to read {file_name}: {e}")

if len(meg_series_list) == 0:
    raise ValueError("No valid Meg sequence was found")

# Use the first file as the reference sequence.
reference_series = meg_series_list[0]

# Calculate the FastDTW distance from each sequence to the reference.
distances = []
for i, series in enumerate(meg_series_list):
    try:
        dist, _ = fastdtw(reference_series, series, dist=lambda x, y: abs(x - y))
        distances.append(dist)
    except Exception as e:
        print(f"[error] failed to calculate the DTW distance for {file_names[i]}: {e}")
        distances.append(np.inf)

# Treat distances above the mean plus one standard deviation as outliers.
valid_distances = [d for d in distances if np.isfinite(d)]
mean_dist = np.mean(valid_distances)
std_dist = np.std(valid_distances)
threshold = mean_dist + std_dist

print(f"\nMean DTW distance: {mean_dist:.2f}, threshold: {threshold:.2f}")
print("==== Potential outlier files ====")
for file, dist in zip(file_names, distances):
    if dist > threshold:
        print(f"{file} -> DTW distance: {dist:.2f}")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import umap.umap_ as umap
from dtaidistance import dtw
from dtaidistance import dtw_ndim
import Data_processing
from pathlib import Path

matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# Detect within-cluster outliers using DTW.
def detect_outliers_dtw(data_df, labels, threshold=2.0):
    """
    data_df: Feature data containing reduced magnetic-field segments.
    labels: Cluster labels.
    threshold: Mark distances above mean + threshold * standard deviation.
    """
    outlier_indices = []
    unique_labels = [l for l in np.unique(labels) if l != -1]

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue

        avg_dists = np.zeros(len(idx))
        for i, id_i in enumerate(idx):
            s1 = data_df.iloc[id_i].values.astype(np.double)
            dists = [
                dtw.distance_fast(s1, data_df.iloc[id_j].values.astype(np.double))
                for id_j in idx if id_j != id_i
            ]
            avg_dists[i] = np.mean(dists)

        mean_dist, std_dist = np.mean(avg_dists), np.std(avg_dists)
        for i, sample_id in enumerate(idx):
            if avg_dists[i] > mean_dist + threshold * std_dist:
                outlier_indices.append(sample_id)
                print(f"Outlier in cluster {label}: index={sample_id}, mean distance={avg_dists[i]:.4f}")

    return outlier_indices


# Step 1: Load and preprocess the clustered data.
SCRIPT_DIR = Path(__file__).resolve().parent
cluster_file = SCRIPT_DIR / "anchor_cluster" / "anchor_cluster_route_004.csv"
refined_file = SCRIPT_DIR / "anchor_cluster" / "anchor_cluster_route_004_refined.csv"
data = pd.read_csv(cluster_file)
data = Data_processing.Data_Preprocessing(data)

# Extract magnetic-field signal segments.
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# Preprocess magnetic-field data, including smoothing and normalization.
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front,
                                                                            columns_meg_back)

meg_front_pred = pd.DataFrame(meg_front_pred).interpolate(axis=1).fillna(0)
meg_back_pred = pd.DataFrame(meg_back_pred).interpolate(axis=1).fillna(0)

# Step 2: Reduce the pre-anchor and post-anchor segments separately.
umap_model_front = umap.UMAP(n_components=20, random_state=42)
meg_front_umap = pd.DataFrame(umap_model_front.fit_transform(meg_front_pred), index=meg_front_pred.index)

umap_model_back = umap.UMAP(n_components=20, random_state=42)
meg_back_umap = pd.DataFrame(umap_model_back.fit_transform(meg_back_pred), index=meg_back_pred.index)

# Step 3: Detect and remove within-cluster outliers.
labels = data.iloc[:, -1].values  # The final column contains Cluster_Label.

# Detect outliers in pre-anchor segments.
outliers_front = detect_outliers_dtw(meg_front_umap, labels, threshold=2.0)
# Detect outliers in post-anchor segments.
outliers_back = detect_outliers_dtw(meg_back_umap, labels, threshold=2.0)

# Combine the detected outlier indices.
outliers_total = set(outliers_front).union(set(outliers_back))
print(f"Total outliers: {len(outliers_total)}")

# Remove outliers.
data_cleaned = data.drop(index=outliers_total).reset_index(drop=True)

# Save the cleaned data.
data_cleaned.to_csv(refined_file, index=False)
print("Cleaned data saved.")

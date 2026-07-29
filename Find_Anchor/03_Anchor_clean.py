import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import Data_processing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm
from pathlib import Path
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False



# Calculate the pairwise DTW distance matrix.
def compute_dtw_distance_matrix(data_df):
    data = data_df.values
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    # Avoid calculating symmetric entries twice.
    for i in tqdm(range(n_samples), desc="Calculating the DTW distance matrix"):
        for j in range(i + 1, n_samples):
            dist = dtw.distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


# Step 1: Load and preprocess the data.
SCRIPT_DIR = Path(__file__).resolve().parent
anchor_file = SCRIPT_DIR / "anchor" / "anchor_combined_route_004.csv"
data = pd.read_csv(anchor_file)

# Magnetic-field signal segments.
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# Preprocess the signal segments.
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)

meg_pred = pd.DataFrame(meg_pred)

# Reduce the dimensionality of the magnetic-field data.
pca = PCA(n_components=20)
meg_pca = pd.DataFrame(pca.fit_transform(meg_pred), index=meg_pred.index)

# Step 2: Calculate the distance matrix.
meg_dist = compute_dtw_distance_matrix(meg_pca)

min_distances = []
for i in range(len(meg_dist)):
    # Exclude the self-distance before finding the minimum.
    dists = meg_dist[i].copy()
    dists[i] = np.inf
    min_distances.append(np.min(dists))

min_distances = np.array(min_distances)

# Set the distance threshold using the 75th percentile.
threshold = np.percentile(min_distances, 75)

# Identify pseudo-anchor indices.
pseudo_anchor_indices = np.where(min_distances > threshold)[0]

# Report the detected pseudo-anchors.
print(f"Detected pseudo-anchors: {len(pseudo_anchor_indices)}")
print("Pseudo-anchor indices:", pseudo_anchor_indices)


# Step 3: Remove pseudo-anchors and save the result.
cleaned_data = data.drop(index=pseudo_anchor_indices).reset_index(drop=True)

# Overwrite the input file with the cleaned data.
cleaned_data.to_csv(anchor_file, index=False)
print(f"Pseudo-anchors removed. Remaining anchors: {len(cleaned_data)}")

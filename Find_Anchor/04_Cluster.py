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
import umap.umap_ as umap  # Requires the umap-learn package.
import pandas as pd
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
cluster_dir = SCRIPT_DIR / "anchor_cluster"
cluster_dir.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(SCRIPT_DIR / "anchor" / "anchor_combined_route_004.csv")
data = Data_processing.Data_Preprocessing(data)

# Magnetic-field signal segments.
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# Concatenated primary LTE segments.
columns_lte = pd.concat([
    data.iloc[:, 1204:1208],
    data.iloc[:, 1216:1220],
    data.iloc[:, 1228:1232]
], axis=1)

# LTE segments before and after each anchor.
columns_lte_front = pd.concat([
    data.iloc[:, 1200:1204],
    data.iloc[:, 1212:1216],
    data.iloc[:, 1224:1228]
], axis=1)

columns_lte_back = pd.concat([
    data.iloc[:, 1208:1212],
    data.iloc[:, 1220:1224],
    data.iloc[:, 1232:1236]
], axis=1)

# Preprocess the signal segments.
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
lte_pred = Data_processing.LTE_Preprocessing(columns_lte) * 4
lte_front_pred = Data_processing.LTE_Preprocessing(columns_lte_front)
lte_back_pred = Data_processing.LTE_Preprocessing(columns_lte_back)

meg_pred = pd.DataFrame(meg_pred)
meg_front_pred = pd.DataFrame(meg_front_pred)
meg_back_pred = pd.DataFrame(meg_back_pred)

# Combine the extracted features.
meg_combined = pd.concat([meg_pred, meg_front_pred, meg_back_pred], axis=1)
meg_combined = meg_combined.interpolate(axis=1).fillna(0)

lte_combined = pd.concat([lte_pred, lte_front_pred, lte_back_pred], axis=1)
lte_combined=lte_combined.fillna(0)

# Reduce the dimensionality of the magnetic-field data.
# pca = PCA(n_components=50)
# meg_pca = pd.DataFrame(pca.fit_transform(meg_pred), index=meg_pred.index)
umap_model = umap.UMAP(n_components=20, random_state=42)
meg_umap = pd.DataFrame(umap_model.fit_transform(meg_pred), index=meg_pred.index)

# Step 2: Calculate distance matrices.
meg_dist = compute_dtw_distance_matrix(meg_umap)
lte_dist = pairwise_distances(lte_combined, metric='euclidean')
# lte_dist = compute_dtw_distance_matrix(lte_combined)

# Step 3: Normalize and combine the distance matrices.
scaler = MinMaxScaler()
meg_dist_norm = scaler.fit_transform(meg_dist)
lte_dist_norm = scaler.fit_transform(lte_dist)

alpha = 0.3
D_total = np.sqrt((alpha * meg_dist_norm) ** 2 + ((1 - alpha) * lte_dist_norm) ** 2)
np.fill_diagonal(D_total, 0)
# Step 4: Cluster the combined distance matrix.
# clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5)
# labels = clusterer.fit_predict(D_total)
# data['Cluster_Label'] = labels

n_clusters = 10  # Adjust for the target dataset if necessary.
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(D_total)
data['Cluster_Label'] = labels

# Save the clustered data.
data.to_csv(cluster_dir / "anchor_cluster_route_004.csv", index=False)

# Step 5: Evaluate clustering quality.
score = silhouette_score(D_total, labels, metric='precomputed')
#
# mask = labels != -1
# score = silhouette_score(meg_dist_norm[mask][:, mask], labels[mask], metric='precomputed')
print(f"Silhouette Score: {score:.4f}")

# Step 6: Visualize the clusters.
pca_2d = PCA(n_components=2)
pca_features_2d = pca_2d.fit_transform(meg_combined)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_features_2d[:, 0], pca_features_2d[:, 1], c=labels, cmap='tab20', s=15)
plt.title('Two-dimensional visualization of fused-distance clusters (KMeans)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

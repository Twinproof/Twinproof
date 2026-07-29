import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import Data_processing
from pathlib import Path



def load_anchor_class(file_path):
    data = pd.read_csv(file_path)
    columns_meg = data.iloc[:, :400]
    columns_meg_front = data.iloc[:, 400:800]
    columns_meg_back = data.iloc[:, 800:1200]

    columns_lte = pd.concat([
        data.iloc[:, 1204:1208],
        data.iloc[:, 1216:1220],
        data.iloc[:, 1228:1232]
    ], axis=1)

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

    meg_pred, _, _ = Data_processing.Meg_Preprocessing(
        columns_meg, columns_meg_front, columns_meg_back
    )
    lte_pred = Data_processing.LTE_Preprocessing(columns_lte) * 4
    lte_front_pred = Data_processing.LTE_Preprocessing(columns_lte_front)
    lte_back_pred = Data_processing.LTE_Preprocessing(columns_lte_back)
    lte_combined = pd.concat([lte_pred, lte_front_pred, lte_back_pred], axis=1)
    lte_combined = lte_combined.fillna(0)

    meg_pred = pd.DataFrame(meg_pred)
    return meg_pred, lte_combined



def average_set_euclidean_pca(df_A, df_B, n_components=20):
    n_comp = min(n_components, df_A.shape[0], df_B.shape[0])
    pca = PCA(n_components=n_comp)
    arr_A = pca.fit_transform(df_A)
    arr_B = pca.fit_transform(df_B)
    distances = [euclidean(a, b) for a in arr_A for b in arr_B]
    return np.mean(distances)

def average_set_euclidean(df_A, df_B):
    arr_A = df_A.to_numpy()
    arr_B = df_B.to_numpy()
    distances = [euclidean(a, b) for a in arr_A for b in arr_B]
    return np.mean(distances)



def same(feature_folder, new_index, k=3.0, alpha=0.5):
    """Process the inputs and return the corresponding result."""
    
    files = sorted([f for f in os.listdir(feature_folder) if f.endswith(".csv")])
    
    indices = [int(f.split("_")[-1].split(".")[0]) for f in files]

    
    old_indices = [idx for idx in indices if idx < new_index]
    new_indices = [idx for idx in indices if idx >= new_index]

    all_pairs = []

    for new_idx in new_indices:
        new_file = os.path.join(feature_folder, f"anchor_feature_{new_idx}.csv")
        meg_new, lte_new = load_anchor_class(new_file)

        distances = []
        for old_idx in old_indices:
            old_file = os.path.join(feature_folder, f"anchor_feature_{old_idx}.csv")
            meg_old, lte_old = load_anchor_class(old_file)

            
            meg_dist = average_set_euclidean_pca(meg_new, meg_old)
            lte_dist = average_set_euclidean(lte_new, lte_old)
            combined_dist = alpha * meg_dist + (1 - alpha) * lte_dist
            distances.append((old_idx, combined_dist))

        
        dists = np.array([d for _, d in distances])
        norm_dist = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)

        
        mean_dist = norm_dist.mean()
        std_dist = norm_dist.std()
        threshold = max(mean_dist - k * std_dist, 0)

        
        for (old_idx, _), dist in zip(distances, norm_dist):
            if dist < threshold:
                all_pairs.append([old_idx, new_idx])

    return all_pairs




feature_folder = Path(__file__).resolve().parent / "Anchor_feature_parking"
new_index = 4  
pairs = same(feature_folder, new_index, k=3.0)
print("Similar anchor pairs: ", pairs)

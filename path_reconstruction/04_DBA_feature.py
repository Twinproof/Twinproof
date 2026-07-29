import pandas as pd
import numpy as np
import os
from dtaidistance import dtw
from sklearn.decomposition import PCA
import Data_processing
from pathlib import Path


def reduce_dimension(data, n_components=50):
    """Process the inputs and return the corresponding result."""
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    max_components = min(n_components, data.shape[0], data.shape[1])  
    pca = PCA(n_components=max_components)
    reduced = pca.fit_transform(data)
    return reduced


def find_medoid_sequence(sequences):
    """Process the inputs and return the corresponding result."""
    n = sequences.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw.distance(sequences[i], sequences[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    total_distances = distance_matrix.sum(axis=1)
    medoid_index = np.argmin(total_distances)
    return medoid_index


def process_anchor_file(file_path, label):
    """Process the inputs and return the corresponding result."""
    data = pd.read_csv(file_path)

    
    columns_meg = data.iloc[:, :400]  # (n_samples, 400)
    columns_meg_front = data.iloc[:, 400:800]
    columns_meg_back = data.iloc[:, 800:1200]
    columns_lte = pd.concat([
        data.iloc[:, 1204:1208],
        data.iloc[:, 1216:1220],
        data.iloc[:, 1228:1232]
    ], axis=1)

    
    meg_pred, _, _ = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
    lte_pred = Data_processing.LTE_Preprocessing(columns_lte) * 4

    
    meg_pred = pd.DataFrame(meg_pred)
    lte_pred = pd.DataFrame(lte_pred)
    meg_reduced = reduce_dimension(meg_pred, n_components=30)
    idx_meg = find_medoid_sequence(meg_reduced)
    idx_lte = find_medoid_sequence(lte_pred.to_numpy())

    
    meg_row = columns_meg.iloc[idx_meg].to_numpy()
    lte_row = columns_lte.iloc[idx_lte].to_numpy()
    combined = np.concatenate([meg_row, lte_row, [int(label)]])  
    return combined, idx_meg, idx_lte


def process_all_anchors(feature_folder, output_file="Anchor_features_DBA.csv"):
    """Process the inputs and return the corresponding result."""
    all_data = []
    files = sorted([f for f in os.listdir(feature_folder) if f.startswith("anchor_feature_") and f.endswith(".csv")])

    for f in files:
        label = int(f.split("_")[-1].split(".")[0])
        file_path = os.path.join(feature_folder, f)
        print(f"Processing {file_path} (label={label}) ...")

        combined, idx_meg, idx_lte = process_anchor_file(file_path, label)
        print(f"  -> MEG center row: {idx_meg}, LTE center row: {idx_lte}")
        all_data.append(combined)

    
    all_data = np.vstack(all_data)
    columns = [f"meg_{i}" for i in range(400)] + [f"lte_{i+1}" for i in range(12)] + ["label"]
    df_out = pd.DataFrame(all_data, columns=columns)
    df_out["label"] = df_out["label"].astype(int)
    df_out.to_csv(output_file, index=False)
    print(f"All anchor features saved to {output_file}")



if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    feature_folder = script_dir / "Anchor_feature_parking"
    process_all_anchors(feature_folder, script_dir / "Anchor_features_DBA.csv")

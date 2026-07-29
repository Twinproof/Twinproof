import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Global_Anchor_Labeling import (
    CURRENT_COLUMNS,
    FEATURE_COLUMNS,
    dtw_distance,
    pca_reduce,
    preprocess_lte_frame,
    preprocess_meg_frame,
    split_04_cluster_features,
)


def feature_id_from_path(path):
    match = re.search(r"anchor_feature_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse global anchor id from file name: {path.name}")
    return int(match.group(1))


def find_default_anchor_table(anchor_dir):
    candidates = sorted(anchor_dir.glob("*_global.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_global.csv file found in {anchor_dir}")
    if len(candidates) > 1:
        print(f"[warn] multiple global anchor tables found, using: {candidates[0]}")
    return candidates[0]


def load_template_rows(feature_dir):
    rows = []
    labels = []
    for feature_path in sorted(feature_dir.glob("anchor_feature_*.csv")):
        global_id = feature_id_from_path(feature_path)
        df = pd.read_csv(feature_path, usecols=FEATURE_COLUMNS, encoding="utf-8-sig")
        rows.append(df)
        labels.extend([global_id] * len(df))

    if not rows:
        raise FileNotFoundError(f"No anchor_feature_*.csv files found in {feature_dir}")
    return pd.concat(rows, ignore_index=True), np.asarray(labels, dtype=int)


def standardize_columns(values):
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


def preprocess_signal_features(template_features, local_features, meg_components):
    """
    Build comparable feature spaces for templates and local anchors.

    MEG follows the 04_Cluster.py-style preprocessing and is reduced from the
    middle 400 dimensions to about 50 dimensions. LTE uses the same RSSI slices
    used by Global_Anchor_Labeling.py, then column-standardizes them so they can
    be concatenated with the reduced MEG features for S_signal.
    """
    all_features = pd.concat(
        [template_features[FEATURE_COLUMNS], local_features[FEATURE_COLUMNS]],
        ignore_index=True,
    )
    meg, _, _, lte, lte_front, lte_back = split_04_cluster_features(all_features)
    meg_processed = preprocess_meg_frame(meg)
    meg_reduced = pca_reduce(meg_processed, n_components=meg_components)

    lte_pred = preprocess_lte_frame(lte) * 4
    lte_front_pred = preprocess_lte_frame(lte_front)
    lte_back_pred = preprocess_lte_frame(lte_back)
    lte_combined = pd.concat([lte_pred, lte_front_pred, lte_back_pred], axis=1).fillna(0.0).to_numpy(dtype=float)

    meg_reduced = standardize_columns(meg_reduced)
    lte_combined = standardize_columns(lte_combined)
    joint_features = np.concatenate([meg_reduced, lte_combined], axis=1)

    n_template = len(template_features)
    return {
        "template_meg": meg_reduced[:n_template],
        "local_meg": meg_reduced[n_template:],
        "template_lte": lte_combined[:n_template],
        "local_lte": lte_combined[n_template:],
        "template_joint": joint_features[:n_template],
        "local_joint": joint_features[n_template:],
    }


def euclidean_distance(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    dim = max(len(left), 1)
    return float(np.linalg.norm(left - right) / np.sqrt(dim))


def similarity_from_distance(distance):
    if not np.isfinite(distance):
        return 0.0
    return float(1.0 / (1.0 + max(distance, 0.0)))


def mean_template_similarity(local_vector, template_vectors, distance_fn):
    if len(template_vectors) == 0:
        return np.nan
    scores = [similarity_from_distance(distance_fn(local_vector, template)) for template in template_vectors]
    return float(np.mean(scores))


def calculate_signal_scores(local_table, template_features, template_labels, feature_spaces):
    template_indices_by_label = {
        int(label): np.where(template_labels == label)[0]
        for label in np.unique(template_labels)
    }

    output_rows = []
    for local_index, row in local_table.reset_index(drop=True).iterrows():
        global_id_value = row.get("Global_Anchor_ID")
        if pd.isna(global_id_value):
            template_indices = np.array([], dtype=int)
            global_id = np.nan
        else:
            global_id = int(global_id_value)
            template_indices = template_indices_by_label.get(global_id, np.array([], dtype=int))

        s_meg = mean_template_similarity(
            feature_spaces["local_meg"][local_index],
            feature_spaces["template_meg"][template_indices],
            dtw_distance,
        )
        s_lte = mean_template_similarity(
            feature_spaces["local_lte"][local_index],
            feature_spaces["template_lte"][template_indices],
            euclidean_distance,
        )
        s_signal = mean_template_similarity(
            feature_spaces["local_joint"][local_index],
            feature_spaces["template_joint"][template_indices],
            dtw_distance,
        )

        output_rows.append({
            "File_Name": row.get("File_Name"),
            "Mid_Time": row.get("Mid_Time"),
            "Anchor_Info": row.get("Anchor_Info"),
            "Global_Anchor_ID": global_id,
            "S_lte": round(s_lte, 6) if pd.notna(s_lte) else np.nan,
            "S_meg": round(s_meg, 6) if pd.notna(s_meg) else np.nan,
            "S_signal": round(s_signal, 6) if pd.notna(s_signal) else np.nan,
        })

    return pd.DataFrame(output_rows)


def build_signal_score_table(anchor_table, feature_dir, output_path, meg_components):
    local_table = pd.read_csv(anchor_table, encoding="utf-8-sig")
    required_columns = ["File_Name", "Mid_Time", "Anchor_Info", "Global_Anchor_ID"]
    missing = [col for col in required_columns if col not in local_table.columns]
    if missing:
        raise ValueError(f"Anchor table missing columns: {missing}")

    local_table = local_table.dropna(subset=["Global_Anchor_ID"]).reset_index(drop=True)
    template_features, template_labels = load_template_rows(feature_dir)
    feature_spaces = preprocess_signal_features(template_features, local_table, meg_components)
    scores = calculate_signal_scores(local_table, template_features, template_labels, feature_spaces)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False, encoding="utf-8-sig")
    return scores


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_anchor_dir = project_root / "Claim_Detection" / "anchor_with_global_id"

    parser = argparse.ArgumentParser(
        description="Calculate MEG, LTE, and concatenated signal similarity scores for matched global anchors."
    )
    parser.add_argument("--anchor-table", type=Path, default=None)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=project_root / "path_reconstruction" / "Anchor_feature_parking",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "signal" / "forged_trace" / "scores.csv",
    )
    parser.add_argument("--meg-components", type=int, default=50)
    args = parser.parse_args()

    anchor_table = args.anchor_table or find_default_anchor_table(default_anchor_dir)
    scores = build_signal_score_table(anchor_table, args.feature_dir, args.output, args.meg_components)

    print(f"[done] wrote {args.output}")
    print(f"[done] rows: {len(scores)}")
    if not scores.empty:
        print(scores.head().to_string(index=False))


if __name__ == "__main__":
    main()

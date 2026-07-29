import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 400
CURRENT_COLUMNS = [f"Column_{index}" for index in range(1, WINDOW_SIZE + 1)]
PRE_COLUMNS = [f"Pre_Anchor_{index}" for index in range(1, WINDOW_SIZE + 1)]
POST_COLUMNS = [f"Post_Anchor_{index}" for index in range(1, WINDOW_SIZE + 1)]
RSSI_COLUMNS = [f"Cell_RSSI_{cell}_{index}" for cell in range(1, 4) for index in range(1, 13)]
ID_COLUMNS = [f"Cell_ID_{cell}_{index}" for cell in range(1, 4) for index in range(1, 13)]
FEATURE_COLUMNS = CURRENT_COLUMNS + PRE_COLUMNS + POST_COLUMNS + RSSI_COLUMNS + ID_COLUMNS


def claim_path_to_merged_path(claim_path, data_root):
    """Map Claim_Path to the corresponding merged data file."""
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")

    folder = parts[0]
    stem = parts[-1]
    if stem.endswith("_merged.csv"):
        file_name = stem
    elif stem.endswith("_merged"):
        file_name = f"{stem}.csv"
    else:
        file_name = f"{stem}_merged.csv"
    return data_root / folder / file_name


def unique_claim_paths(claims_file):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    return list(dict.fromkeys(claims["Claim_Path"].astype(str).tolist()))


def iter_claim_paths(claims_file, unique=False):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    if unique:
        for claim_path in unique_claim_paths(claims_file):
            yield None, claim_path
    else:
        for claim_index, claim_path in enumerate(claims["Claim_Path"].astype(str)):
            yield claim_index, claim_path


def row_numeric(values, fill_value=0.0):
    series = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    return series.ffill().bfill().fillna(fill_value).to_numpy()


def row_zscore(values):
    arr = row_numeric(values)
    std = arr.std()
    if std == 0:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def smooth_row(values, window=5):
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def preprocess_meg_frame(df):
    """
    04_cluster.py calls Data_processing.Meg_Preprocessing.
    This local version keeps the same intent: numeric fill, smoothing, row-wise z-score.
    """
    processed = []
    for _, row in df.iterrows():
        processed.append(row_zscore(smooth_row(row_numeric(row.values))))
    return pd.DataFrame(processed, index=df.index)


def preprocess_lte_frame(df):
    """
    04_cluster.py calls Data_processing.LTE_Preprocessing.
    This local version performs numeric fill and light smoothing.
    """
    arr = []
    for _, row in df.iterrows():
        arr.append(smooth_row(row_numeric(row.values, fill_value=0.0), window=3))
    return pd.DataFrame(arr, index=df.index).fillna(0.0)


def pca_reduce(train_and_query, n_components=20):
    """
    UMAP is unavailable in the bundled runtime, so PCA/SVD is used as a deterministic fallback.
    The important part is preserved: all global samples and local samples share one reduced space.
    """
    x = np.asarray(train_and_query, dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    max_components = max(1, min(n_components, x.shape[0], x.shape[1]))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:max_components].T


def dtw_distance(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0

    for left_value in left:
        current = np.full(len(right) + 1, np.inf)
        for j, right_value in enumerate(right, start=1):
            cost = abs(left_value - right_value)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current

    return float(previous[-1] / max(len(left) + len(right), 1))


def minmax(values):
    arr = np.asarray(values, dtype=float)
    low = arr.min()
    high = arr.max()
    if high == low:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def local_peaks(values, distance=50, mode="max"):
    """Small peak/valley detector used to reproduce the 02_Anchor_find anchor windows."""
    arr = np.asarray(values, dtype=float)
    candidates = []
    for index in range(1, len(arr) - 1):
        if mode == "max" and arr[index] >= arr[index - 1] and arr[index] >= arr[index + 1]:
            candidates.append(index)
        if mode == "min" and arr[index] <= arr[index - 1] and arr[index] <= arr[index + 1]:
            candidates.append(index)

    selected = []
    for index in sorted(candidates, key=lambda i: abs(arr[i]), reverse=True):
        if all(abs(index - old) >= distance for old in selected):
            selected.append(index)
    return np.array(sorted(selected), dtype=int)


def find_anchor_segments(data, window_size=WINDOW_SIZE, step_size=10):
    """Detect anchor segments from merged data, following Find_Anchor/02_Anchor_find.py."""
    data = data.copy()

    for col in ["Cell_RSSI_1", "Cell_RSSI_2", "Cell_RSSI_3"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").ffill().bfill()
        data[col] = data[col].rolling(window=5, min_periods=1).mean()

    for col in ["Cell_ID_1", "Cell_ID_2", "Cell_ID_3"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").ffill().bfill()

    meg = pd.to_numeric(data["Meg"], errors="coerce").ffill().bfill()
    meg_diff = meg.diff().rolling(window=5, min_periods=1).mean().fillna(0)
    meg_diff_scaled = row_zscore(meg_diff)
    meg_diff_abs = np.abs(meg_diff_scaled)

    upper_bound_meg = meg_diff_abs.mean() + 0.08 * meg_diff_abs.std()
    peaks = local_peaks(meg_diff_scaled, distance=50, mode="max")
    valleys = local_peaks(meg_diff_scaled, distance=50, mode="min")
    peak_values = meg_diff_scaled[peaks] if len(peaks) else np.array([0.0])
    valley_values = meg_diff_scaled[valleys] if len(valleys) else np.array([0.0])
    upper_bound_peaks = peak_values.mean() + 0.6 * peak_values.std()
    under_bound_valley = valley_values.mean() - valley_values.std()

    anchor_points = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_scaled = meg_diff_scaled[start:end]
        window_peaks = local_peaks(window_scaled, distance=50, mode="max")
        window_valleys = local_peaks(window_scaled, distance=50, mode="min")

        mean_meg_diff = meg_diff_abs[start:end].mean()
        max_peak = window_scaled[window_peaks].max() if len(window_peaks) else 0
        min_valley = window_scaled[window_valleys].min() if len(window_valleys) else 0
        rolling_std = pd.Series(window_scaled).rolling(window=20, min_periods=1).std().mean()

        lte_rssi_std = np.mean([
            data[col].iloc[start:end].std()
            for col in ["Cell_RSSI_1", "Cell_RSSI_2", "Cell_RSSI_3"]
        ])

        id_changes = 0
        for col in ["Cell_ID_1", "Cell_ID_2", "Cell_ID_3"]:
            ids = data[col].iloc[start:end].fillna(-1).astype(int).values
            id_changes += np.count_nonzero(np.diff(ids) != 0)
        lte_id_change_rate = id_changes / (3 * (window_size - 1))

        meg_flag = (
            mean_meg_diff >= upper_bound_meg
            and max_peak >= upper_bound_peaks
            and min_valley <= under_bound_valley
            and rolling_std >= 0.08
        )
        lte_flag = lte_rssi_std > 2.0 or lte_id_change_rate > 0.02

        if meg_flag or lte_flag:
            anchor_points.append((start, end))

    return merge_anchor_segments(anchor_points, meg_diff_abs, window_size)


def merge_anchor_segments(anchor_points, meg_diff_abs, window_size):
    merged_segments = []
    for segment in sorted(anchor_points):
        if not merged_segments or merged_segments[-1][1] < segment[0]:
            merged_segments.append(segment)
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], segment[1]))

    final_segments = []
    for start, end in merged_segments:
        if end - start > window_size:
            max_index = int(np.argmax(meg_diff_abs[start:end])) + start
            new_start = max(max_index - window_size // 2, start)
            new_end = min(new_start + window_size, end)
            final_segments.append((new_start, new_end))
        else:
            final_segments.append((start, end))

    filtered_segments = []
    index = 0
    while index < len(final_segments) - 1:
        current_start, current_end = final_segments[index]
        next_start, next_end = final_segments[index + 1]
        if next_start - current_end <= 600:
            combined_start = current_start
            combined_end = next_end
            max_index = int(np.argmax(meg_diff_abs[combined_start:combined_end])) + combined_start
            new_start = max(max_index - window_size // 2, combined_start)
            new_end = min(new_start + window_size, combined_end)
            filtered_segments.append((new_start, new_end))
            index += 2
        else:
            filtered_segments.append((current_start, current_end))
            index += 1
    if index == len(final_segments) - 1:
        filtered_segments.append(final_segments[index])

    return filtered_segments


def pad_to_window(values, pad_front=False):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    if len(arr) >= WINDOW_SIZE:
        return arr[-WINDOW_SIZE:] if pad_front else arr[:WINDOW_SIZE]
    pad_width = WINDOW_SIZE - len(arr)
    if pad_front:
        return np.pad(arr, (pad_width, 0), constant_values=np.nan)
    return np.pad(arr, (0, pad_width), constant_values=np.nan)


def format_elapsed_time(seconds):
    seconds = max(0, int(round(float(seconds))))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def anchor_mid_time(data, start, end):
    mid_index = min(max((start + end) // 2, 0), len(data) - 1)
    if "Time" not in data.columns:
        return format_elapsed_time(mid_index)

    times = pd.to_datetime(data["Time"], errors="coerce")
    if times.isna().all() or pd.isna(times.iloc[mid_index]):
        return format_elapsed_time(mid_index)

    elapsed = (times.iloc[mid_index] - times.dropna().iloc[0]).total_seconds()
    return format_elapsed_time(elapsed)


def extract_anchor_feature_row(data, start, end, file_name):
    """Build one anchor-style feature row from a detected segment."""
    current = pad_to_window(data["Meg"].iloc[start:end], pad_front=False)
    pre = pad_to_window(data["Meg"].iloc[max(0, start - WINDOW_SIZE):start], pad_front=True)
    post = pad_to_window(data["Meg"].iloc[end:end + WINDOW_SIZE], pad_front=False)

    row = {col: current[index] for index, col in enumerate(CURRENT_COLUMNS)}
    row.update({col: pre[index] for index, col in enumerate(PRE_COLUMNS)})
    row.update({col: post[index] for index, col in enumerate(POST_COLUMNS)})

    for cell in range(1, 4):
        rssi = pad_to_window(data[f"Cell_RSSI_{cell}"].iloc[max(0, start - 4):min(len(data), end + 4)], pad_front=False)[:12]
        cell_id = pad_to_window(data[f"Cell_ID_{cell}"].iloc[max(0, start - 4):min(len(data), end + 4)], pad_front=False)[:12]
        for index in range(12):
            row[f"Cell_RSSI_{cell}_{index + 1}"] = rssi[index]
            row[f"Cell_ID_{cell}_{index + 1}"] = cell_id[index]

    row["File_Name"] = file_name
    row["Mid_Time"] = anchor_mid_time(data, start, end)
    row["Anchor_Info"] = f"({start},{end})"
    return row


def feature_id_from_path(path):
    match = re.search(r"anchor_feature_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse global anchor id from file name: {path.name}")
    return int(match.group(1))


def load_global_rows(feature_dir, max_samples_per_anchor):
    rows = []
    labels = []
    for feature_path in sorted(feature_dir.glob("anchor_feature_*.csv")):
        global_id = feature_id_from_path(feature_path)
        df = pd.read_csv(feature_path, usecols=FEATURE_COLUMNS, encoding="utf-8-sig")
        if max_samples_per_anchor and len(df) > max_samples_per_anchor:
            df = df.sample(n=max_samples_per_anchor, random_state=42)
        rows.append(df)
        labels.extend([global_id] * len(df))

    if not rows:
        raise FileNotFoundError(f"No anchor_feature_*.csv files found in {feature_dir}")
    return pd.concat(rows, ignore_index=True), np.array(labels, dtype=int)


def build_local_rows(claims_file, data_root, unique=False):
    rows = []
    for claim_index, claim_path in iter_claim_paths(claims_file, unique=unique):
        merged_path = claim_path_to_merged_path(claim_path, data_root)
        if not merged_path.exists():
            print(f"[skip] merged file not found: {merged_path}")
            continue

        data = pd.read_csv(merged_path, encoding="utf-8-sig")
        segments = find_anchor_segments(data)
        prefix = f"claim row {claim_index}" if claim_index is not None else "unique claim"
        print(f"[anchors] {prefix} {merged_path.name}: {segments}")

        for start, end in segments:
            row = extract_anchor_feature_row(data, start, end, merged_path.name)
            row["Claim_Row_Index"] = claim_index
            row["Claim_Path"] = claim_path
            rows.append(row)
    return pd.DataFrame(rows)


def split_04_cluster_features(df):
    """Replicate 04_cluster.py Step 1 column slicing by column names."""
    meg = df[CURRENT_COLUMNS]
    meg_front = df[PRE_COLUMNS]
    meg_back = df[POST_COLUMNS]

    lte = pd.concat([
        df[[f"Cell_RSSI_{cell}_{i}" for i in range(5, 9)]]
        for cell in range(1, 4)
    ], axis=1)
    lte_front = pd.concat([
        df[[f"Cell_RSSI_{cell}_{i}" for i in range(1, 5)]]
        for cell in range(1, 4)
    ], axis=1)
    lte_back = pd.concat([
        df[[f"Cell_RSSI_{cell}_{i}" for i in range(9, 13)]]
        for cell in range(1, 4)
    ], axis=1)
    return meg, meg_front, meg_back, lte, lte_front, lte_back


def preprocess_04_cluster_style(all_features):
    """
    Match 04_cluster.py Step 1:
    - preprocess MEG current/front/back
    - preprocess LTE current/front/back
    - reduce MEG current 400 dimensions
    - concatenate LTE current/front/back
    """
    meg, meg_front, meg_back, lte, lte_front, lte_back = split_04_cluster_features(all_features)

    meg_pred = preprocess_meg_frame(meg)
    _ = preprocess_meg_frame(meg_front)
    _ = preprocess_meg_frame(meg_back)

    lte_pred = preprocess_lte_frame(lte) * 4
    lte_front_pred = preprocess_lte_frame(lte_front)
    lte_back_pred = preprocess_lte_frame(lte_back)
    lte_combined = pd.concat([lte_pred, lte_front_pred, lte_back_pred], axis=1).fillna(0.0)

    meg_reduced = pca_reduce(meg_pred, n_components=20)
    return meg_reduced, lte_combined.to_numpy(dtype=float)


def match_local_to_global(global_features, global_labels, local_features, alpha, top_k):
    all_features = pd.concat([global_features, local_features[FEATURE_COLUMNS]], ignore_index=True)
    meg_reduced, lte_combined = preprocess_04_cluster_style(all_features)

    n_global = len(global_features)
    global_meg = meg_reduced[:n_global]
    local_meg = meg_reduced[n_global:]
    global_lte = lte_combined[:n_global]
    local_lte = lte_combined[n_global:]

    matches = []
    for local_index in range(len(local_features)):
        meg_distances = np.array([dtw_distance(local_meg[local_index], g) for g in global_meg])
        lte_distances = np.linalg.norm(global_lte - local_lte[local_index], axis=1)

        meg_norm = minmax(meg_distances)
        lte_norm = minmax(lte_distances)
        fused = np.sqrt((alpha * meg_norm) ** 2 + ((1 - alpha) * lte_norm) ** 2)

        by_label = {}
        for label in np.unique(global_labels):
            label_distances = np.sort(fused[global_labels == label])
            by_label[int(label)] = float(np.mean(label_distances[:top_k]))

        best_label = min(by_label, key=by_label.get)
        matches.append({
            "Global_Anchor_ID": best_label,
            "Global_Match_Distance": round(by_label[best_label], 6),
            "MEG_Distance": round(float(meg_distances[fused.argmin()]), 6),
            "LTE_Distance": round(float(lte_distances[fused.argmin()]), 6),
        })

    return pd.DataFrame(matches)


def build_global_anchor_table(claims_file, data_root, feature_dir, output_path, alpha, top_k, max_samples_per_anchor, unique=False):
    global_features, global_labels = load_global_rows(feature_dir, max_samples_per_anchor)
    local_features = build_local_rows(claims_file, data_root, unique=unique)
    if local_features.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        local_features.to_csv(output_path, index=False, encoding="utf-8-sig")
        return local_features

    matches = match_local_to_global(global_features, global_labels, local_features, alpha, top_k)
    result = pd.concat([local_features.reset_index(drop=True), matches], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def load_local_anchor_files(anchor_files):
    rows = []
    for anchor_file in anchor_files:
        df = pd.read_csv(anchor_file, encoding="utf-8-sig")
        missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{anchor_file} missing feature columns: {missing[:10]}")
        rows.append(df)
        print(f"[load] {anchor_file}: {len(df)} rows")

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def build_global_anchor_table_from_anchor_files(anchor_files, feature_dir, output_path, alpha, top_k, max_samples_per_anchor):
    global_features, global_labels = load_global_rows(feature_dir, max_samples_per_anchor)
    local_features = load_local_anchor_files(anchor_files)
    if local_features.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        local_features.to_csv(output_path, index=False, encoding="utf-8-sig")
        return local_features

    matches = match_local_to_global(global_features, global_labels, local_features, alpha, top_k)
    result = pd.concat([local_features.reset_index(drop=True), matches], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Detect anchors from claim paths and match them to global anchor ids.")
    parser.add_argument("--claims", type=Path, default=project_root / "Claim" / "forged_trace_claims.csv")
    parser.add_argument("--data-root", type=Path, default=project_root / "data" / "collectionData_02")
    parser.add_argument(
        "--local-anchor-files",
        type=Path,
        nargs="+",
        default=None,
        help="Use existing anchor csv files directly instead of detecting anchors from Claim_Path.",
    )
    parser.add_argument("--feature-dir", type=Path, default=project_root / "path_reconstruction" / "Anchor_feature_parking")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Claim_Detection" / "anchor_with_global_id" / "anchor_combined_route_004_global.csv",
    )
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples-per-anchor", type=int, default=30)
    parser.add_argument(
        "--unique-claim-paths",
        action="store_true",
        help="Deduplicate Claim_Path values before detecting anchors.",
    )
    args = parser.parse_args()

    if args.local_anchor_files:
        result = build_global_anchor_table_from_anchor_files(
            anchor_files=args.local_anchor_files,
            feature_dir=args.feature_dir,
            output_path=args.output,
            alpha=args.alpha,
            top_k=args.top_k,
            max_samples_per_anchor=args.max_samples_per_anchor,
            unique=args.unique_claim_paths,
        )
    else:
        result = build_global_anchor_table(
            claims_file=args.claims,
            data_root=args.data_root,
            feature_dir=args.feature_dir,
            output_path=args.output,
            alpha=args.alpha,
            top_k=args.top_k,
            max_samples_per_anchor=args.max_samples_per_anchor,
        )

    print(f"Saved labeled anchor table to: {args.output}")
    if result.empty:
        print("No anchors were detected.")
    else:
        print("Global_Anchor_ID distribution:")
        print(result["Global_Anchor_ID"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

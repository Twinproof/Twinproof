import argparse
import ast
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from Global_Anchor_Labeling import (  # noqa: E402
    FEATURE_COLUMNS,
    extract_anchor_feature_row,
    load_global_rows,
    match_local_to_global,
)


SENSOR_WINDOW_ROWS = 400
SENSOR_TAIL_ROWS = 800
SIGNAL_WINDOW_ROWS = 8
TOPO_JUMP_THRESHOLD = 4


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (SyntaxError, ValueError):
        return []


def first_n(values, n, fill_value=np.nan):
    values = list(values)[:n]
    if len(values) < n:
        values.extend([fill_value] * (n - len(values)))
    return values


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def expand_signal_to_sensor_rows(signal_df, sensor_len):
    expanded = pd.DataFrame(index=range(sensor_len))
    cell_ids = []
    cell_rssis = []

    for _, row in signal_df.iterrows():
        ids = first_n(parse_list(row.get("Cell_ID")), 3)
        rssis = first_n(parse_list(row.get("Cell_RSSI")), 3)
        cell_ids.append(ids)
        cell_rssis.append(rssis)

    if not cell_ids:
        cell_ids = [[np.nan, np.nan, np.nan]]
        cell_rssis = [[np.nan, np.nan, np.nan]]

    for sensor_start in range(sensor_len):
        signal_index = min(sensor_start // 100, len(cell_ids) - 1)
        ids = cell_ids[signal_index]
        rssis = cell_rssis[signal_index]
        for cell in range(1, 4):
            expanded.loc[sensor_start, f"Cell_ID_{cell}"] = ids[cell - 1]
            expanded.loc[sensor_start, f"Cell_RSSI_{cell}"] = rssis[cell - 1]

    return expanded.ffill().bfill()


def build_merged_sensor_signal(sensor_path, signal_path):
    sensor_df = read_csv(sensor_path)
    signal_df = read_csv(signal_path)

    ore_col = find_ore_column(sensor_df)
    merged = pd.DataFrame({
        "Time": sensor_df["Time"] if "Time" in sensor_df.columns else np.arange(len(sensor_df)),
        "Meg": sensor_df["Meg"],
        "Ore": sensor_df[ore_col],
    })
    signal_expanded = expand_signal_to_sensor_rows(signal_df, len(sensor_df))
    merged = pd.concat([merged, signal_expanded], axis=1)

    for col in ["Meg", "Ore", "Cell_ID_1", "Cell_ID_2", "Cell_ID_3", "Cell_RSSI_1", "Cell_RSSI_2", "Cell_RSSI_3"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill().bfill()
    return merged


def find_ore_column(df):
    for col in df.columns:
        if col.lower() == "ore":
            return col
    raise ValueError("sensor file does not contain ore/Ore column")


def circular_abs_diff(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def circular_mean(values):
    radians = np.deg2rad(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float))
    if len(radians) == 0:
        return np.nan
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()
    return float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)


def ore_transition_score(before_ore, current_ore):
    before = pd.to_numeric(pd.Series(before_ore), errors="coerce").ffill().bfill().to_numpy(dtype=float)
    current = pd.to_numeric(pd.Series(current_ore), errors="coerce").ffill().bfill().to_numpy(dtype=float)
    n = min(len(before), len(current))
    if n == 0:
        return 0.0

    before = before[-n:]
    current = current[:n]
    profile_jump = float(np.nanmean(circular_abs_diff(before, current)) / 180.0)
    boundary_jump = float(circular_abs_diff([before[-1]], [current[0]])[0] / 180.0)
    mean_jump = float(circular_abs_diff([circular_mean(before)], [circular_mean(current)])[0] / 180.0)
    abnormal = 0.5 * boundary_jump + 0.3 * mean_jump + 0.2 * profile_jump
    return float(np.clip(1.0 - abnormal, 0.0, 1.0))


def random_ore_score(sensor_df, rng, samples=2, window_rows=SENSOR_WINDOW_ROWS):
    valid_starts = range(window_rows, max(window_rows, len(sensor_df) - window_rows + 1))
    if len(sensor_df) < window_rows * 2 or not valid_starts:
        return np.nan, []

    starts = [rng.randint(window_rows, len(sensor_df) - window_rows) for _ in range(samples)]
    scores = []
    for start in starts:
        before = sensor_df["Ore"].iloc[start - window_rows:start]
        current = sensor_df["Ore"].iloc[start:start + window_rows]
        scores.append(ore_transition_score(before, current))
    return float(np.mean(scores)), starts


def tail_ore_score(sensor_df):
    if len(sensor_df) < SENSOR_TAIL_ROWS + SENSOR_WINDOW_ROWS:
        return np.nan
    before = sensor_df["Ore"].iloc[-SENSOR_TAIL_ROWS - SENSOR_WINDOW_ROWS:-SENSOR_TAIL_ROWS]
    current = sensor_df["Ore"].iloc[-SENSOR_TAIL_ROWS:-SENSOR_TAIL_ROWS + SENSOR_WINDOW_ROWS]
    return ore_transition_score(before, current)


def parse_connected(value):
    parsed = parse_list(value)
    return [int(item) for item in parsed if pd.notna(item)]


def load_topology_graph(connection_file):
    df = pd.read_csv(connection_file, encoding="utf-8-sig")
    graph = {}
    for _, row in df.iterrows():
        node = int(row["Cluster_Label"])
        neighbors = parse_connected(row["Connected_Classes"])
        graph.setdefault(node, set()).update(neighbors)
        for neighbor in neighbors:
            graph.setdefault(neighbor, set()).add(node)
    return graph


def shortest_path_distance(graph, start, end):
    if start == end:
        return 0
    if start not in graph or end not in graph:
        return None

    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def topo_break_score(distance, threshold):
    if distance is None:
        return 0.0
    if distance <= threshold:
        return 1.0
    return float((threshold / distance) ** 2)


def paired_signal_file(sensor_path):
    signal_name = sensor_path.name.replace("sensor_", "signal_", 1)
    signal_path = sensor_path.with_name(signal_name)
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found for {sensor_path}: {signal_path}")
    return signal_path


def collect_trace_transplant_items(data_root, folder_markers):
    items = []
    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if not any(marker in folder.name for marker in folder_markers):
            continue
        for sensor_path in sorted(folder.glob("sensor_*.csv")):
            items.append({
                "file_name": f"{folder.name}/{sensor_path.name}",
                "sensor_path": sensor_path,
                "signal_path": paired_signal_file(sensor_path),
            })
    return items


def transplant_claim_path_to_file_name(claim_path):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")
    folder = parts[0]
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    if stem.startswith("signal_"):
        stem = stem[len("signal_"):]
    return f"{folder}/sensor_{stem}.csv"


def collect_trace_transplant_claim_items(claims_file, data_root):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    items = []
    for claim_index, claim_path in enumerate(claims["Claim_Path"].astype(str)):
        file_name = transplant_claim_path_to_file_name(claim_path)
        folder, sensor_name = file_name.replace("\\", "/").split("/", 1)
        sensor_path = data_root / folder / sensor_name
        signal_path = sensor_path.with_name(sensor_name.replace("sensor_", "signal_", 1))
        if not sensor_path.exists() or not signal_path.exists():
            print(f"[skip] raw sensor/signal not found for claim row {claim_index}: {file_name}")
            continue
        items.append({
            "file_name": file_name,
            "sensor_path": sensor_path,
            "signal_path": signal_path,
        })
    return items


def claim_path_to_file_name(claim_path):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")
    folder = parts[0]
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    return f"{folder}/{stem}_merged.csv"


def file_name_to_sensor_signal_paths(file_name, raw_data_root):
    folder, merged_name = file_name.replace("\\", "/").split("/", 1)
    stem = merged_name[:-11] if merged_name.endswith("_merged.csv") else Path(merged_name).stem
    source_dir = raw_data_root / folder
    return source_dir / f"sensor_{stem}.csv", source_dir / f"signal_{stem}.csv"


def collect_forged_trace_items(claims_file, raw_data_root):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    items = []
    seen = set()
    for claim_path in claims["Claim_Path"].astype(str):
        file_name = claim_path_to_file_name(claim_path)
        if file_name in seen:
            continue
        seen.add(file_name)

        sensor_path, signal_path = file_name_to_sensor_signal_paths(file_name, raw_data_root)
        if not sensor_path.exists() or not signal_path.exists():
            print(f"[skip] raw sensor/signal not found for {file_name}")
            continue
        items.append({
            "file_name": file_name,
            "sensor_path": sensor_path,
            "signal_path": signal_path,
        })
    return items


def replay_claim_path_to_stem(claim_path):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    if stem.startswith("signal_"):
        stem = stem[len("signal_"):]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    return stem


def collect_replay_trace_items(claims_file, replay_folder):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    items = []
    for claim_index, claim_path in enumerate(claims["Claim_Path"].astype(str)):
        stem = replay_claim_path_to_stem(claim_path)
        base_name = f"claim_{claim_index:03d}_{stem}"
        sensor_path = replay_folder / f"sensor_{base_name}.csv"
        signal_path = replay_folder / f"signal_{base_name}.csv"
        file_name = f"{replay_folder.name}/sensor_{base_name}.csv"

        if not sensor_path.exists() or not signal_path.exists():
            print(f"[skip] replay sensor/signal not found for claim row {claim_index}: {file_name}")
            continue
        items.append({
            "file_name": file_name,
            "sensor_path": sensor_path,
            "signal_path": signal_path,
        })
    return items


def build_break_feature_rows(items, rng):
    metadata = []
    feature_rows = []
    merged_cache = {}

    for sample_index, item in enumerate(items):
        sensor_path = item["sensor_path"]
        signal_path = item["signal_path"]
        merged = build_merged_sensor_signal(sensor_path, signal_path)
        merged_cache[sample_index] = merged

        if len(merged) < SENSOR_WINDOW_ROWS * 2:
            metadata.append({
                "Sample_ID": sample_index,
                "file_name": item["file_name"],
                "Break_Start": np.nan,
                "Status": "too_short",
            })
            continue

        start = rng.randint(SENSOR_WINDOW_ROWS, len(merged) - SENSOR_WINDOW_ROWS)
        pairs = [
            ("previous", start - SENSOR_WINDOW_ROWS, start),
            ("current", start, start + SENSOR_WINDOW_ROWS),
        ]
        for role, seg_start, seg_end in pairs:
            row = extract_anchor_feature_row(merged, seg_start, seg_end, sensor_path.name)
            row["Sample_ID"] = sample_index
            row["Window_Role"] = role
            feature_rows.append(row)

        metadata.append({
            "Sample_ID": sample_index,
            "file_name": item["file_name"],
            "Break_Start": start,
            "Status": "ok",
        })

    return pd.DataFrame(feature_rows), pd.DataFrame(metadata), merged_cache


def calculate_current_scores(items, feature_dir, connection_file, seed, max_samples_per_anchor, alpha, top_k):
    rng = random.Random(seed)
    graph = load_topology_graph(connection_file)
    feature_rows, metadata, merged_cache = build_break_feature_rows(items, rng)

    match_lookup = {}
    if not feature_rows.empty:
        global_features, global_labels = load_global_rows(feature_dir, max_samples_per_anchor)
        matches = match_local_to_global(global_features, global_labels, feature_rows, alpha=alpha, top_k=top_k)
        matched_rows = pd.concat([feature_rows[["Sample_ID", "Window_Role"]].reset_index(drop=True), matches], axis=1)
        for _, row in matched_rows.iterrows():
            match_lookup[(int(row["Sample_ID"]), row["Window_Role"])] = int(row["Global_Anchor_ID"])

    rows = []
    for _, meta in metadata.iterrows():
        sample_id = int(meta["Sample_ID"])
        merged = merged_cache.get(sample_id)

        if merged is None:
            s_random, random_starts = np.nan, []
            s_tail = np.nan
        else:
            s_random, random_starts = random_ore_score(merged, rng)
            s_tail = tail_ore_score(merged)

        if pd.isna(s_random) and pd.isna(s_tail):
            s_post = np.nan
        elif pd.isna(s_random):
            s_post = s_tail
        elif pd.isna(s_tail):
            s_post = s_random
        else:
            s_post = 0.2 * s_random + 0.8 * s_tail

        prev_node = match_lookup.get((sample_id, "previous"))
        curr_node = match_lookup.get((sample_id, "current"))
        distance = shortest_path_distance(graph, prev_node, curr_node) if prev_node is not None and curr_node is not None else None
        reach_flag = 1 if distance is not None else -1
        s_topo_jump = topo_break_score(distance, TOPO_JUMP_THRESHOLD)

        s_current = np.nan
        if reach_flag == -1:
            s_current = -1.0
        elif pd.notna(s_post):
            s_current = 0.8 * s_post + 0.2 * s_topo_jump

        rows.append({
            "file_name": meta["file_name"],
            "S_random": round(float(s_random), 6) if pd.notna(s_random) else np.nan,
            "S_tail": round(float(s_tail), 6) if pd.notna(s_tail) else np.nan,
            "S_post_check": round(float(s_post), 6) if pd.notna(s_post) else np.nan,
            "Random_Starts": json.dumps(random_starts),
            "Break_Start": meta["Break_Start"],
            "Prev_Global_Anchor_ID": prev_node,
            "Curr_Global_Anchor_ID": curr_node,
            "Reach_Flag": reach_flag,
            "Topo_Distance": distance if distance is not None else np.nan,
            "S_topo_jump": round(float(s_topo_jump), 6),
            "S_current": round(float(s_current), 6) if pd.notna(s_current) else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate current validation scores.")
    parser.add_argument(
        "--mode",
        choices=["trace_transplant", "forged_trace", "replay_trace", "all"],
        default="all",
        help="Attack type to calculate. Default calculates both trace_transplant and forged_trace.",
    )
    parser.add_argument("--transplant-root", type=Path, default=project_root / "data" / "collectionData")
    parser.add_argument("--transplant-folder-markers", nargs="+", default=["route_006", "route_007"])
    parser.add_argument("--transplant-claims-file", type=Path, default=project_root / "Claim" / "transplant_trace_claims.csv")
    parser.add_argument("--claims-file", type=Path, default=project_root / "Claim" / "forged_trace_claims.csv")
    parser.add_argument("--replay-claims-file", type=Path, default=project_root / "Claim" / "replay_trace_claims.csv")
    parser.add_argument(
        "--replay-folder",
        type=Path,
        default=project_root / "data" / "collectionData" / "route_007",
    )
    parser.add_argument("--raw-data-root", type=Path, default=project_root / "data" / "collectionData")
    parser.add_argument("--feature-dir", type=Path, default=project_root / "path_reconstruction" / "Anchor_feature_parking")
    parser.add_argument("--connection-file", type=Path, default=project_root / "path_reconstruction" / "Anchor_connection.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-anchor", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    jobs = []
    if args.mode in ("trace_transplant", "all"):
        output = args.output if args.output and args.mode == "trace_transplant" else (
            project_root / "Claim_Detection" / "results" / "current_validation" / "trace_transplant" / "scores.csv"
        )
        jobs.append((
            "trace_transplant",
            collect_trace_transplant_claim_items(args.transplant_claims_file, args.transplant_root),
            output,
        ))

    if args.mode in ("forged_trace", "all"):
        output = args.output if args.output and args.mode == "forged_trace" else (
            project_root / "Claim_Detection" / "results" / "current_validation" / "forged_trace" / "scores.csv"
        )
        jobs.append((
            "forged_trace",
            collect_forged_trace_items(args.claims_file, args.raw_data_root),
            output,
        ))

    if args.mode in ("replay_trace", "all"):
        output = args.output if args.output and args.mode == "replay_trace" else (
            project_root / "Claim_Detection" / "results" / "current_validation" / "replay_trace" / "scores.csv"
        )
        jobs.append((
            "replay_trace",
            collect_replay_trace_items(args.replay_claims_file, args.replay_folder),
            output,
        ))

    for name, items, output in jobs:
        if not items:
            raise FileNotFoundError(f"No input files found for {name}")

        scores = calculate_current_scores(
            items=items,
            feature_dir=args.feature_dir,
            connection_file=args.connection_file,
            seed=args.seed,
            max_samples_per_anchor=args.max_samples_per_anchor,
            alpha=args.alpha,
            top_k=args.top_k,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"[done] {name}: wrote {output}")
        print(f"[done] {name}: rows {len(scores)}")
        print(scores.head().to_string(index=False))


if __name__ == "__main__":
    main()

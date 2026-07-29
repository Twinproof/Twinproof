import argparse
import heapq
from pathlib import Path

import pandas as pd

try:
    from .PDR_Path import calculate_pdr_length_from_csv
except ImportError:
    from PDR_Path import calculate_pdr_length_from_csv


def file_name_to_sensor_path(file_name, sensor_dir):
    stem = Path(file_name).stem
    if stem.endswith("_merged"):
        stem = stem[:-len("_merged")]
    return sensor_dir / f"sensor_{stem}.csv"


def build_graph(paths_csv):
    paths = pd.read_csv(paths_csv)
    graph = {}
    for _, row in paths.iterrows():
        start = int(row["Start_Anchor"])
        end = int(row["End_Anchor"])
        length = float(row["Path_Length"])
        graph.setdefault(start, []).append((end, length))
        graph.setdefault(end, []).append((start, length))
    return graph


def shortest_distance(graph, start, end):
    start = int(start)
    end = int(end)
    if start == end:
        return 0.0

    queue = [(0.0, start)]
    best = {start: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == end:
            return distance
        if distance > best.get(node, float("inf")):
            continue
        for next_node, edge_length in graph.get(node, []):
            new_distance = distance + edge_length
            if new_distance < best.get(next_node, float("inf")):
                best[next_node] = new_distance
                heapq.heappush(queue, (new_distance, next_node))

    raise ValueError(f"No topology path between anchors {start} and {end}")


def topology_length_for_nodes(graph, nodes):
    if len(nodes) < 2:
        return 0.0

    total = 0.0
    for start, end in zip(nodes, nodes[1:]):
        total += shortest_distance(graph, start, end)
    return total


def length_score(pdr_length, topo_length):
    denominator = max(float(pdr_length), float(topo_length), 1e-9)
    relative_error = abs(float(pdr_length) - float(topo_length)) / denominator
    return max(0.0, min(1.0, 1.0 - relative_error))


def build_scores(anchor_csv, paths_csv, sensor_dir):
    anchors = pd.read_csv(anchor_csv)
    required = {"File_Name", "Mid_Time", "Global_Anchor_ID"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"Missing required columns in anchor csv: {sorted(missing)}")

    graph = build_graph(paths_csv)
    results = []

    for file_name, group in anchors.groupby("File_Name", sort=True):
        group = group.sort_index()
        if len(group) < 2:
            continue

        start_time = group.iloc[0]["Mid_Time"]
        end_time = group.iloc[-1]["Mid_Time"]
        nodes = [int(node) for node in group["Global_Anchor_ID"].tolist()]

        sensor_path = file_name_to_sensor_path(file_name, sensor_dir)
        if not sensor_path.exists():
            raise FileNotFoundError(f"Sensor file not found for {file_name}: {sensor_path}")

        pdr_length = calculate_pdr_length_from_csv(sensor_path, start_time=start_time, end_time=end_time)
        topo_length = topology_length_for_nodes(graph, nodes)
        score = length_score(pdr_length, topo_length)

        results.append({
            "File_Name": file_name,
            "S_Length": round(score, 6),
        })

    return pd.DataFrame(results, columns=["File_Name", "S_Length"])


def _find_default_anchor_csv(base_dir):
    candidates = sorted((base_dir / "anchor_with_global_id").glob("*route_004*_global.csv"))
    if not candidates:
        raise FileNotFoundError("No default route_004 global anchor CSV found.")
    return candidates[0]


def _find_default_sensor_dir(project_dir):
    collection_dir = project_dir / "data" / "collectionData"
    for sensor_dir in collection_dir.iterdir():
        if sensor_dir.is_dir() and (sensor_dir / "sensor_20230628_1436.csv").exists():
            return sensor_dir
    raise FileNotFoundError("No default sensor directory found.")


def main():
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    default_anchor_csv = _find_default_anchor_csv(base_dir)
    default_paths_csv = base_dir / "Paths.csv"
    default_sensor_dir = _find_default_sensor_dir(project_dir)
    default_output = base_dir / "length_score.csv"

    parser = argparse.ArgumentParser(description="Calculate topology length consistency scores.")
    parser.add_argument("--anchor-csv", type=Path, default=default_anchor_csv)
    parser.add_argument("--paths-csv", type=Path, default=default_paths_csv)
    parser.add_argument("--sensor-dir", type=Path, default=default_sensor_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    scores = build_scores(args.anchor_csv, args.paths_csv, args.sensor_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(scores)} scores to {args.output}")


if __name__ == "__main__":
    main()

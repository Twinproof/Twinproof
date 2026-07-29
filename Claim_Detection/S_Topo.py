import argparse
import ast
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


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


def parse_connected(value):
    return [int(item) for item in parse_list(value) if pd.notna(item)]


def parse_regions(value):
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def claim_path_to_file_name(claim_path):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    return f"{stem}_merged.csv"


def find_default_anchor_table(anchor_dir):
    candidates = sorted(anchor_dir.glob("*_global.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_global.csv file found in {anchor_dir}")
    if len(candidates) > 1:
        print(f"[warn] multiple global anchor tables found, using {candidates[0]}")
    return candidates[0]


def load_topology(connection_file):
    df = pd.read_csv(connection_file, encoding="utf-8-sig")
    graph = {}
    node_regions = {}

    for _, row in df.iterrows():
        node = int(row["Cluster_Label"])
        neighbors = parse_connected(row["Connected_Classes"])
        graph.setdefault(node, set()).update(neighbors)
        for neighbor in neighbors:
            graph.setdefault(neighbor, set()).add(node)
        node_regions[node] = parse_regions(row.get("Environment_Constraint"))

    return graph, node_regions


def shortest_path(graph, start, end):
    if start == end:
        return [start]
    if start not in graph or end not in graph:
        return None

    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for neighbor in sorted(graph.get(node, [])):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def expand_reachable_path(node_seq, graph):
    if not node_seq:
        return [], True
    expanded = [node_seq[0]]
    for left, right in zip(node_seq, node_seq[1:]):
        path = shortest_path(graph, left, right)
        if path is None:
            return expanded, False
        expanded.extend(path[1:])
    return expanded, True


def regions_for_path(node_path, node_regions):
    return [node_regions.get(int(node), []) for node in node_path]


def region_match(claim_region, detected_candidates):
    return claim_region in set(detected_candidates)


def lcs_candidate_score(claim_regions, detected_region_candidates):
    if not claim_regions and not detected_region_candidates:
        return 1.0
    if not claim_regions or not detected_region_candidates:
        return 0.0

    rows = len(claim_regions)
    cols = len(detected_region_candidates)
    dp = np.zeros((rows + 1, cols + 1), dtype=int)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if region_match(claim_regions[i - 1], detected_region_candidates[j - 1]):
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return float(dp[rows, cols] / max(rows, cols))


def whole_region_score(claim_regions, detected_region_candidates):
    claim_set = set(claim_regions)
    detected_set = {region for candidates in detected_region_candidates for region in candidates}
    if not claim_set and not detected_set:
        return 1.0
    union = claim_set | detected_set
    if not union:
        return 0.0
    return float(len(claim_set & detected_set) / len(union))


def build_anchor_lookup(anchor_table):
    anchors = pd.read_csv(anchor_table, encoding="utf-8-sig")
    required = {"File_Name", "Global_Anchor_ID"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"Anchor table missing columns: {sorted(missing)}")

    anchors = anchors.dropna(subset=["File_Name", "Global_Anchor_ID"]).copy()
    anchors["Global_Anchor_ID"] = anchors["Global_Anchor_ID"].astype(int)
    lookup = {}
    if "Claim_Row_Index" in anchors.columns:
        anchors = anchors.dropna(subset=["Claim_Row_Index"]).copy()
        anchors["Claim_Row_Index"] = anchors["Claim_Row_Index"].astype(int)
        for claim_index, group in anchors.groupby("Claim_Row_Index", sort=False):
            lookup[int(claim_index)] = {
                "matched_file_name": str(group["File_Name"].iloc[0]),
                "nodes": group["Global_Anchor_ID"].tolist(),
            }
        return lookup, "claim_row_index"

    for file_name, group in anchors.groupby("File_Name", sort=False):
        lookup[str(file_name)] = {
            "matched_file_name": str(file_name),
            "nodes": group["Global_Anchor_ID"].tolist(),
        }
    return lookup, "file_name"


def score_one_claim(claim_index, row, anchor_lookup, lookup_mode, graph, node_regions, w_seq, w_whole):
    matched_file_name = claim_path_to_file_name(row["Claim_Path"])
    lookup_key = claim_index if lookup_mode == "claim_row_index" else matched_file_name
    anchor_entry = anchor_lookup.get(lookup_key, {})
    detected_nodes = anchor_entry.get("nodes", [])
    matched_file_name = anchor_entry.get("matched_file_name", matched_file_name)
    reachable_path, reachable = expand_reachable_path(detected_nodes, graph)
    detected_regions = regions_for_path(reachable_path, node_regions)
    claim_trace = parse_list(row["Claim_Trace"])

    if not detected_nodes:
        reach_flag = -1
        s_seq = 0.0
        s_whole = 0.0
        s_topo = 0.0
        decision = "no_detected_anchor"
    elif not reachable:
        reach_flag = -1
        s_seq = 0.0
        s_whole = 0.0
        s_topo = 0.0
        decision = "unreachable_attack"
    else:
        reach_flag = 1
        s_seq = lcs_candidate_score(claim_trace, detected_regions)
        s_whole = whole_region_score(claim_trace, detected_regions)
        s_topo = w_seq * s_seq + w_whole * s_whole
        decision = "reachable_scored"

    return {
        "Claim_Path": row.get("Claim_Path"),
        "Claim_Time": row.get("Claim_Time"),
        "Claim_Trace": row.get("Claim_Trace"),
        "Label_Time": row.get("Label_Time"),
        "Label_Trace": row.get("Label_Trace"),
        "Attack_Type": row.get("Attack_Type"),
        "Matched_File_Name": matched_file_name,
        "Detected_Global_Node_Seq": detected_nodes,
        "Reachable_Global_Node_Path": reachable_path,
        "Detected_Region_Candidates": detected_regions,
        "Reach_Flag": reach_flag,
        "S_seq": round(float(s_seq), 6),
        "S_whole": round(float(s_whole), 6),
        "S_topo": round(float(s_topo), 6),
        "Topo_Decision": decision,
    }


def build_topology_scores(claims_file, anchor_table, connection_file, output_dir, w_seq, w_whole):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    anchor_lookup, lookup_mode = build_anchor_lookup(anchor_table)
    graph, node_regions = load_topology(connection_file)

    rows = [
        score_one_claim(claim_index, row, anchor_lookup, lookup_mode, graph, node_regions, w_seq, w_whole)
        for claim_index, row in claims.iterrows()
    ]
    scores = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    return scores


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate topology consistency scores for forged trace claims.")
    parser.add_argument("--claims", type=Path, default=project_root / "Claim" / "forged_trace_claims.csv")
    parser.add_argument("--anchor-table", type=Path, default=None)
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "anchor_with_global_id",
    )
    parser.add_argument(
        "--connection-file",
        type=Path,
        default=project_root / "path_reconstruction" / "Anchor_connection.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "topology" / "forged_trace",
    )
    parser.add_argument("--w-seq", type=float, default=0.5)
    parser.add_argument("--w-whole", type=float, default=0.5)
    args = parser.parse_args()

    anchor_table = args.anchor_table or find_default_anchor_table(args.anchor_dir)
    scores = build_topology_scores(
        claims_file=args.claims,
        anchor_table=anchor_table,
        connection_file=args.connection_file,
        output_dir=args.output_dir,
        w_seq=args.w_seq,
        w_whole=args.w_whole,
    )

    print(f"[done] wrote {args.output_dir / 'scores.csv'}")
    print(f"[done] rows: {len(scores)}")


if __name__ == "__main__":
    main()

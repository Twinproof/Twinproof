import argparse
import ast
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ATTACK_CONFIGS = [
    ("forged_trace", "Claim/forged_trace_claims.csv"),
    ("replay_trace", "Claim/replay_trace_claims.csv"),
    ("transplant_trace", "Claim/transplant_trace_claims.csv"),
]

CELL_SIZE_METERS = 10.0
MAX_WALKING_SPEED = 2.0
TIME_TAU = 0.25


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_claim_path(value):
    parts = str(value).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        return "", str(value).strip()
    return parts[-2], parts[-1]


def merged_file_name(stem):
    name = str(stem).strip()
    if name.endswith(".csv"):
        return name
    if name.endswith("_merged"):
        return f"{name}.csv"
    return f"{name}_merged.csv"


def build_merged_index(data_root):
    by_folder_and_name = {}
    by_name = {}
    for path in data_root.rglob("*_merged.csv"):
        by_folder_and_name[(path.parent.name, path.name)] = path
        by_name.setdefault(path.name, []).append(path)
    return by_folder_and_name, by_name


def resolve_merged_path(path_value, data_root, by_folder_and_name, by_name):
    folder, stem = normalize_claim_path(path_value)
    file_name = merged_file_name(stem)

    direct = data_root / folder / file_name
    if direct.exists():
        return direct

    for subdir in data_root.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / folder / file_name
        if candidate.exists():
            return candidate

    indexed = by_folder_and_name.get((folder, file_name))
    if indexed is not None:
        return indexed

    candidates = by_name.get(file_name, [])
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(f"Cannot resolve merged file for {path_value}")


def submitted_path_for_row(row):
    attack_path = row.get("Attack_Path")
    if pd.notna(attack_path) and str(attack_path).strip():
        return attack_path
    return row.get("Claim_Path")


@lru_cache(maxsize=None)
def load_merged(path_string):
    return pd.read_csv(Path(path_string), encoding="utf-8-sig")


def duration_seconds(df):
    if "Time" not in df.columns or df.empty:
        return float(len(df))

    times = pd.to_datetime(df["Time"], errors="coerce").dropna()
    if len(times) < 2:
        return float(len(df))

    duration = (times.iloc[-1] - times.iloc[0]).total_seconds()
    if duration < 0:
        duration += 24 * 3600
    return max(float(duration), 1.0)


def time_score(claim_time, actual_duration):
    claim_time = pd.to_numeric(pd.Series([claim_time]), errors="coerce").iloc[0]
    if pd.isna(claim_time) or actual_duration <= 0:
        return 0.0
    relative_error = abs(float(claim_time) - actual_duration) / max(actual_duration, 1.0)
    return float(math.exp(-relative_error / TIME_TAU))


def parse_region(region):
    match = re.fullmatch(r"([a-zA-Z])(\d+)", str(region).strip())
    if not match:
        return None
    row = ord(match.group(1).lower()) - ord("a")
    col = int(match.group(2)) - 1
    return row, col


def route_jumps(trace):
    points = [parse_region(item) for item in trace]
    jumps = []
    for left, right in zip(points, points[1:]):
        if left is None or right is None:
            jumps.append(None)
        else:
            jumps.append(abs(left[0] - right[0]) + abs(left[1] - right[1]))
    return jumps


def route_score(trace):
    jumps = route_jumps(trace)
    valid_jumps = [jump for jump in jumps if jump is not None]
    if not jumps:
        return 1.0
    if not valid_jumps:
        return 0.0

    penalties = []
    for jump in valid_jumps:
        if jump <= 1:
            penalties.append(0.0)
        elif jump == 2:
            penalties.append(0.25)
        else:
            penalties.append(min(1.0, 0.25 + 0.25 * (jump - 2)))

    invalid_ratio = (len(jumps) - len(valid_jumps)) / max(len(jumps), 1)
    score = 1.0 - float(np.mean(penalties)) - 0.5 * invalid_ratio
    return float(np.clip(score, 0.0, 1.0))


def speed_score(trace, claim_time):
    claim_time = pd.to_numeric(pd.Series([claim_time]), errors="coerce").iloc[0]
    if pd.isna(claim_time) or claim_time <= 0:
        return 0.0

    jumps = [jump for jump in route_jumps(trace) if jump is not None]
    if not jumps:
        return 1.0

    estimated_distance = sum(max(jump, 1) for jump in jumps) * CELL_SIZE_METERS
    estimated_speed = estimated_distance / max(float(claim_time), 1.0)
    if estimated_speed <= MAX_WALKING_SPEED:
        return 1.0
    return float(np.clip(MAX_WALKING_SPEED / estimated_speed, 0.0, 1.0))


def continuity_score(df):
    if "Meg" not in df.columns or df.empty:
        return 0.5

    meg = pd.to_numeric(df["Meg"], errors="coerce")
    missing_ratio = float(meg.isna().mean())
    meg = meg.ffill().bfill().dropna()
    if len(meg) < 3:
        return max(0.0, 1.0 - missing_ratio)

    diff = meg.diff().abs().dropna()
    median_diff = float(diff.median())
    mad = float((diff - median_diff).abs().median())
    active_threshold = max(0.05, median_diff + 0.5 * mad)
    spike_threshold = max(3.0, median_diff + 8.0 * max(mad, 1e-6))

    active_ratio = float((diff > active_threshold).mean())
    spike_ratio = float((diff > spike_threshold).mean())

    if active_ratio < 0.02:
        activity_penalty = 0.4
    elif active_ratio > 0.80:
        activity_penalty = 0.2
    else:
        activity_penalty = 0.0

    score = 1.0 - missing_ratio - activity_penalty - min(0.5, spike_ratio * 5.0)
    return float(np.clip(score, 0.0, 1.0))


def score_one_claim(row, data_root, by_folder_and_name, by_name):
    trace_value = row.get("Claim_Trace", row.get("Claim_Location", ""))
    trace = [str(item) for item in parse_list(trace_value)]
    submitted_path = resolve_merged_path(submitted_path_for_row(row), data_root, by_folder_and_name, by_name)
    df = load_merged(str(submitted_path))

    actual_duration = duration_seconds(df)
    t_score = time_score(row.get("Claim_Time"), actual_duration)
    v_score = speed_score(trace, row.get("Claim_Time"))
    c_score = continuity_score(df)
    r_score = route_score(trace)

    rule_score = (
        0.30 * t_score
        + 0.30 * v_score
        + 0.20 * c_score
        + 0.20 * r_score
    )

    return {
        "Claim_Trace / Claim_Location": trace_value,
        "Matched_File": submitted_path.name,
        "Rule_Score": round(float(rule_score), 6),
        "Time_Check_Score": round(float(t_score), 6),
        "Speed_Check_Score": round(float(v_score), 6),
        "Continuity_Check_Score": round(float(c_score), 6),
        "Route_Check_Score": round(float(r_score), 6),
        "Actual_Duration": round(float(actual_duration), 3),
    }


def score_claims(attack_name, claims_file, data_root, output_dir, by_folder_and_name, by_name):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    rows = []

    for _, row in claims.iterrows():
        try:
            scores = score_one_claim(row, data_root, by_folder_and_name, by_name)
        except Exception as exc:
            print(f"[warn] {attack_name} {row.get('Claim_Path')}: {exc}")
            scores = {
                "Claim_Trace / Claim_Location": row.get("Claim_Trace", row.get("Claim_Location", "")),
                "Matched_File": "",
                "Rule_Score": 0.0,
                "Time_Check_Score": 0.0,
                "Speed_Check_Score": 0.0,
                "Continuity_Check_Score": 0.0,
                "Route_Check_Score": 0.0,
                "Actual_Duration": np.nan,
            }

        rows.append({
            "Attack_Name": attack_name,
            "Claim_Path": row.get("Claim_Path"),
            **scores,
        })

    result = pd.DataFrame(rows)
    attack_dir = output_dir / attack_name
    attack_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(attack_dir / "scores.csv", index=False, encoding="utf-8-sig")
    return result


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Rule-based trajectory plausibility baseline.")
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--claim-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "rule",
    )
    args = parser.parse_args()

    by_folder_and_name, by_name = build_merged_index(args.data_root)
    all_results = []
    for attack_name, relative_path in ATTACK_CONFIGS:
        claims_file = args.claim_root / relative_path
        result = score_claims(attack_name, claims_file, args.data_root, args.output_dir, by_folder_and_name, by_name)
        all_results.append(result)
        print(f"[done] {attack_name}: {len(result)} rows -> {args.output_dir / attack_name / 'scores.csv'}")

    combined = pd.concat(all_results, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "scores.csv", index=False, encoding="utf-8-sig")
    print(f"[done] combined: {len(combined)} rows -> {args.output_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()

import argparse
import re
from pathlib import Path

import pandas as pd


ATTACKS = [
    ("forged_trace", "Forged"),
    ("replay_trace", "Replay"),
    ("trace_transplant", "Transplant"),
]
DEFAULT_THRESHOLDS = [0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
DEFAULT_WEIGHTS = {
    "S_topo": 0.40,
    "S_curr": 0.30,
    "S_signal": 0.20,
    "S_time": 0.10,
}


def sample_key(value):
    """Normalize Claim_Path/File_Name variants to the same sample key."""
    if pd.isna(value):
        return ""
    stem = Path(str(value).strip().replace("\\", "/")).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    stem = re.sub(r"^sensor_claim_\d+_", "", stem)
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    return stem


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def resolve_time_file(results_root, attack_name):
    time_dir = results_root / "time" / attack_name
    for file_name in ("time_score.csv", "scores.csv"):
        candidate = time_dir / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No time score file found in {time_dir}")


def load_topology_scores(results_root, attack_name):
    df = read_csv(results_root / "topology" / attack_name / "scores.csv")
    required = {"Claim_Path", "S_topo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"topology/{attack_name} missing columns: {sorted(missing)}")

    out = df[["Claim_Path", "S_topo"]].copy()
    out["Sample_Key"] = out["Claim_Path"].map(sample_key)
    return out


def load_current_scores(results_root, attack_name):
    df = read_csv(results_root / "current_validation" / attack_name / "scores.csv")
    required = {"file_name", "S_current"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"current_validation/{attack_name} missing columns: {sorted(missing)}")

    df = df[["file_name", "S_current"]].copy()
    df["Sample_Key"] = df["file_name"].map(sample_key)
    grouped = df.groupby("Sample_Key", as_index=False).agg(S_curr=("S_current", "mean"))
    return grouped


def load_signal_scores(results_root, attack_name):
    df = read_csv(results_root / "signal" / attack_name / "scores.csv")
    required = {"File_Name", "S_signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"signal/{attack_name} missing columns: {sorted(missing)}")

    df = df[["File_Name", "S_signal"]].copy()
    df["Sample_Key"] = df["File_Name"].map(sample_key)
    grouped = (
        df.groupby("Sample_Key", as_index=False)
        .agg(S_signal=("S_signal", "mean"), Signal_Anchor_Count=("S_signal", "size"))
    )
    return grouped


def load_time_scores(results_root, attack_name):
    df = read_csv(resolve_time_file(results_root, attack_name))
    required = {"File_Name", "S_Time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"time/{attack_name} missing columns: {sorted(missing)}")

    df = df[["File_Name", "S_Time"]].copy()
    df["Sample_Key"] = df["File_Name"].map(sample_key)
    grouped = df.groupby("Sample_Key", as_index=False).agg(S_time=("S_Time", "mean"))
    return grouped


def build_weighted_scores_for_attack(results_root, attack_name, attack_label, weights):
    base = load_topology_scores(results_root, attack_name)
    current = load_current_scores(results_root, attack_name)
    signal = load_signal_scores(results_root, attack_name)
    time = load_time_scores(results_root, attack_name)

    merged = base.merge(current[["Sample_Key", "S_curr"]], on="Sample_Key", how="left")
    merged = merged.merge(signal, on="Sample_Key", how="left")
    merged = merged.merge(time, on="Sample_Key", how="left")

    for column in ("S_topo", "S_curr", "S_signal", "S_time"):
        merged[f"Missing_{column}"] = merged[column].isna()
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["Attack_Name"] = attack_name
    merged["Attack_Type"] = attack_label
    merged["S_total"] = (
        weights["S_topo"] * merged["S_topo"]
        + weights["S_curr"] * merged["S_curr"]
        + weights["S_signal"] * merged["S_signal"]
        + weights["S_time"] * merged["S_time"]
    )

    columns = [
        "Attack_Name",
        "Attack_Type",
        "Claim_Path",
        "Sample_Key",
        "S_topo",
        "S_curr",
        "S_signal",
        "S_time",
        "S_total",
        "Signal_Anchor_Count",
        "Missing_S_topo",
        "Missing_S_curr",
        "Missing_S_signal",
        "Missing_S_time",
    ]
    return merged[columns]


def build_far_table(weighted_scores, thresholds):
    rows = []
    for threshold in thresholds:
        row = {"Threshold_L": threshold}
        for attack_name, attack_label in ATTACKS:
            attack_scores = weighted_scores[weighted_scores["Attack_Name"] == attack_name]
            if attack_scores.empty:
                row[attack_label] = 0.0
                continue
            row[attack_label] = float((attack_scores["S_total"] >= threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fuse four consistency scores and calculate FAR at multiple thresholds."
    )
    parser.add_argument("--results-root", type=Path, default=project_root / "Claim_Detection" / "results")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Claim_Detection" / "results" / "fusion")
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated thresholds, e.g. 0.55,0.60,0.65,0.70,0.75,0.80.",
    )
    parser.add_argument("--w-topo", type=float, default=DEFAULT_WEIGHTS["S_topo"])
    parser.add_argument("--w-curr", type=float, default=DEFAULT_WEIGHTS["S_curr"])
    parser.add_argument("--w-signal", type=float, default=DEFAULT_WEIGHTS["S_signal"])
    parser.add_argument("--w-time", type=float, default=DEFAULT_WEIGHTS["S_time"])
    args = parser.parse_args()

    weights = {
        "S_topo": args.w_topo,
        "S_curr": args.w_curr,
        "S_signal": args.w_signal,
        "S_time": args.w_time,
    }
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    weighted_scores = pd.concat(
        [
            build_weighted_scores_for_attack(args.results_root, attack_name, attack_label, weights)
            for attack_name, attack_label in ATTACKS
        ],
        ignore_index=True,
    )
    far_table = build_far_table(weighted_scores, args.thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weighted_path = args.output_dir / "weighted_scores.csv"
    far_path = args.output_dir / "far_by_threshold.csv"
    weighted_scores.to_csv(weighted_path, index=False, encoding="utf-8-sig")
    far_table.to_csv(far_path, index=False, encoding="utf-8-sig")

    print(f"[done] weighted scores: {weighted_path}")
    print(f"[done] FAR table: {far_path}")
    print(far_table.to_string(index=False))


if __name__ == "__main__":
    main()

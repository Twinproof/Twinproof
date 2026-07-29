import argparse
from pathlib import Path

import pandas as pd

from S_Fusion_FAR import (
    DEFAULT_THRESHOLDS,
    load_signal_scores,
    load_time_scores,
    load_topology_scores,
)


ATTACKS = ["forged_trace", "replay_trace", "trace_transplant"]
WEIGHTS = {
    "S_topo": 0.40 / 0.70,
    "S_signal": 0.20 / 0.70,
    "S_time": 0.10 / 0.70,
}


def build_scores(results_root, attack_name):
    topology = load_topology_scores(results_root, attack_name)
    signal = load_signal_scores(results_root, attack_name)
    time = load_time_scores(results_root, attack_name)

    merged = topology.merge(signal, on="Sample_Key", how="left")
    merged = merged.merge(time, on="Sample_Key", how="left")

    for column in WEIGHTS:
        merged[f"Missing_{column}"] = merged[column].isna()
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged.insert(0, "Attack_Name", attack_name)
    merged["S_without_current"] = sum(
        weight * merged[column] for column, weight in WEIGHTS.items()
    )
    return merged[
        [
            "Attack_Name",
            "Claim_Path",
            "Sample_Key",
            "S_topo",
            "S_signal",
            "S_time",
            "S_without_current",
            "Signal_Anchor_Count",
            "Missing_S_topo",
            "Missing_S_signal",
            "Missing_S_time",
        ]
    ]


def build_far_table(scores, thresholds):
    rows = []
    for threshold in thresholds:
        row = {"Threshold": threshold}
        for attack_name in ATTACKS:
            attack_scores = scores.loc[
                scores["Attack_Name"] == attack_name, "S_without_current"
            ]
            row[attack_name] = float((attack_scores >= threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Calculate the TwinProof ablation without currency consistency."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "Claim_Detection" / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "Claim_Detection"
        / "results"
        / "ablation"
        / "without_current",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
    )
    args = parser.parse_args()

    all_scores = pd.concat(
        [build_scores(args.results_root, attack_name) for attack_name in ATTACKS],
        ignore_index=True,
    )
    far_table = build_far_table(all_scores, args.thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(
        args.output_dir / "scores.csv", index=False, encoding="utf-8-sig"
    )
    far_table.to_csv(
        args.output_dir / "far_by_threshold.csv", index=False, encoding="utf-8-sig"
    )

    for attack_name in ATTACKS:
        attack_dir = args.output_dir / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        all_scores[all_scores["Attack_Name"] == attack_name].to_csv(
            attack_dir / "scores.csv", index=False, encoding="utf-8-sig"
        )

    print("Weights:", WEIGHTS)
    print(far_table.to_string(index=False))


if __name__ == "__main__":
    main()

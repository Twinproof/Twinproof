import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ATTACKS = {
    "forged_location": "Forged",
    "replay_location": "Replay",
    "transplant_location": "Transplant",
}
DEFAULT_THRESHOLDS = [0.77, 0.78, 0.79, 0.80, 0.81, 0.82]


def load_attack_scores(results_root, attack_name):
    path = results_root / attack_name / "scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing location score file: {path}")

    scores = pd.read_csv(path, encoding="utf-8-sig")
    if "S_location" not in scores.columns:
        raise ValueError(f"{path} missing S_location column")
    scores["Attack_Name"] = attack_name
    return scores


def build_far_rows(scores, thresholds):
    rows = []
    total_count = len(scores)
    for threshold in thresholds:
        accepted = scores["S_location"] >= threshold
        accepted_count = int(accepted.sum())
        rows.append({
            "Attack_Name": scores["Attack_Name"].iloc[0] if total_count else "",
            "Threshold": threshold,
            "Total_Count": total_count,
            "Accepted_Count": accepted_count,
            "FAR": round(accepted_count / total_count, 6) if total_count else 0.0,
        })
    return rows


def build_acceptance_rows(scores, thresholds):
    keep_columns = [
        "Attack_Name",
        "Claim_Row_Index",
        "S_topo",
        "S_lte",
        "S_meg",
        "S_signal",
        "S_location",
    ]
    output = pd.DataFrame()
    for column in keep_columns:
        if column in scores.columns:
            output[column] = scores[column]

    for threshold in thresholds:
        threshold_label = f"{threshold:.2f}"
        output[f"Accepted_at_{threshold_label}"] = scores["S_location"] >= threshold
    return output


def calculate_location_far(results_root, output_dir, thresholds, attacks):
    far_rows = []
    acceptance_tables = []

    for attack_name in attacks:
        scores = load_attack_scores(results_root, attack_name)
        far_rows.extend(build_far_rows(scores, thresholds))
        acceptance_tables.append(build_acceptance_rows(scores, thresholds))

    far_table = pd.DataFrame(far_rows)
    wide_rows = []
    for threshold in thresholds:
        row = {"Threshold_L": threshold}
        for attack_name in attacks:
            attack_label = DEFAULT_ATTACKS.get(attack_name, attack_name)
            matched = far_table[
                (far_table["Attack_Name"] == attack_name)
                & (far_table["Threshold"] == threshold)
            ]
            row[attack_label] = float(matched["FAR"].iloc[0]) if not matched.empty else 0.0
        wide_rows.append(row)
    far_by_threshold = pd.DataFrame(wide_rows)

    acceptance_table = pd.concat(acceptance_tables, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    far_path = output_dir / "location_attack_far.csv"
    far_by_threshold_path = output_dir / "far_by_threshold.csv"
    acceptance_path = output_dir / "location_attack_acceptance.csv"
    far_table.to_csv(far_path, index=False, encoding="utf-8-sig")
    far_by_threshold.to_csv(far_by_threshold_path, index=False, encoding="utf-8-sig")
    acceptance_table.to_csv(acceptance_path, index=False, encoding="utf-8-sig")
    return far_table, far_by_threshold, acceptance_table, far_path, far_by_threshold_path, acceptance_path


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate FAR for location attacks from S_location scores.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "location",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "location" / "far",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--attacks", nargs="+", default=list(DEFAULT_ATTACKS))
    args = parser.parse_args()

    far_table, far_by_threshold, acceptance_table, far_path, far_by_threshold_path, acceptance_path = calculate_location_far(
        results_root=args.results_root,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        attacks=args.attacks,
    )

    print(f"[done] FAR long table: {far_path}")
    print(f"[done] FAR by threshold: {far_by_threshold_path}")
    print(f"[done] acceptance table: {acceptance_path}")
    print(f"[done] acceptance rows: {len(acceptance_table)}")
    print(far_by_threshold.to_string(index=False))


if __name__ == "__main__":
    main()

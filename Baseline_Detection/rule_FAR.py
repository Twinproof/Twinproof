import argparse
from pathlib import Path

import pandas as pd


DEFAULT_THRESHOLDS = [0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
SCORE_COLUMN = "Rule_Score"


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "Claim_Detection" / "results" / "rule"
    default_output = Path(__file__).resolve().parent / "rule_far_results.csv"

    parser = argparse.ArgumentParser(
        description="Calculate FAR for rule scores under multiple thresholds."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing attack subfolders with scores.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="CSV path for saving FAR results.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Threshold values. FAR is the percent of samples with score > threshold.",
    )
    return parser.parse_args()


def find_attack_score_files(input_dir):
    score_files = []
    for attack_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        score_path = attack_dir / "scores.csv"
        if score_path.exists():
            score_files.append((attack_dir.name, score_path))
    return score_files


def calculate_far_for_file(attack_name, score_path, thresholds):
    scores = pd.read_csv(score_path)
    if SCORE_COLUMN not in scores.columns:
        raise ValueError(f"{score_path} does not contain column '{SCORE_COLUMN}'")

    score_values = pd.to_numeric(scores[SCORE_COLUMN], errors="coerce").dropna()
    total = len(score_values)
    if total == 0:
        raise ValueError(f"{score_path} has no valid numeric scores in '{SCORE_COLUMN}'")

    rows = []
    for threshold in thresholds:
        false_accepts = int((score_values > threshold).sum())
        far = false_accepts / total
        rows.append(
            {
                "Attack_Name": attack_name,
                "Threshold": threshold,
                "Total_Samples": total,
                "Accepted_Samples": false_accepts,
                "FAR": far,
                "FAR_Percent": far * 100,
            }
        )
    return rows, score_values


def main():
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    score_files = find_attack_score_files(args.input_dir)
    if not score_files:
        raise FileNotFoundError(f"No attack subfolder scores.csv files found in {args.input_dir}")

    all_rows = []
    all_scores = []
    for attack_name, score_path in score_files:
        rows, score_values = calculate_far_for_file(attack_name, score_path, args.thresholds)
        all_rows.extend(rows)
        all_scores.append(score_values)

    combined_scores = pd.concat(all_scores, ignore_index=True)
    total = len(combined_scores)
    for threshold in args.thresholds:
        false_accepts = int((combined_scores > threshold).sum())
        far = false_accepts / total
        all_rows.append(
            {
                "Attack_Name": "ALL",
                "Threshold": threshold,
                "Total_Samples": total,
                "Accepted_Samples": false_accepts,
                "FAR": far,
                "FAR_Percent": far * 100,
            }
        )

    result = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    display = result.copy()
    display["Threshold"] = display["Threshold"].map(lambda value: f"{value:.2f}")
    display["FAR"] = display["FAR"].map(lambda value: f"{value:.6f}")
    display["FAR_Percent"] = display["FAR_Percent"].map(lambda value: f"{value:.2f}%")
    print(display.to_string(index=False))
    print(f"\nSaved FAR results to: {args.output}")


if __name__ == "__main__":
    main()

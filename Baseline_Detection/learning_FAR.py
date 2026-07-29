import argparse
from pathlib import Path

import pandas as pd


LABEL_COLUMN = "Predicted_Label"


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "Claim_Detection" / "results" / "learning"
    default_output = Path(__file__).resolve().parent / "learning_far_results.csv"

    parser = argparse.ArgumentParser(
        description="Calculate FAR for learning predictions on attack samples."
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
    return parser.parse_args()


def find_attack_score_files(input_dir):
    score_files = []
    for attack_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        score_path = attack_dir / "scores.csv"
        if score_path.exists():
            score_files.append((attack_dir.name, score_path))
    return score_files


def calculate_far_for_file(attack_name, score_path):
    scores = pd.read_csv(score_path)
    if LABEL_COLUMN not in scores.columns:
        raise ValueError(f"{score_path} does not contain column '{LABEL_COLUMN}'")

    labels = pd.to_numeric(scores[LABEL_COLUMN], errors="coerce").dropna().astype(int)
    labels = labels[labels.isin([0, 1])]
    total = len(labels)
    if total == 0:
        raise ValueError(f"{score_path} has no valid 0/1 labels in '{LABEL_COLUMN}'")

    undetected = int((labels == 0).sum())
    detected = int((labels == 1).sum())
    far = undetected / total
    detection_rate = detected / total
    row = {
        "Attack_Name": attack_name,
        "Total_Samples": total,
        "Detected_Samples": detected,
        "Undetected_Samples": undetected,
        "FAR": far,
        "FAR_Percent": far * 100,
        "Detection_Rate": detection_rate,
        "Detection_Rate_Percent": detection_rate * 100,
    }
    return row, labels


def main():
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    score_files = find_attack_score_files(args.input_dir)
    if not score_files:
        raise FileNotFoundError(f"No attack subfolder scores.csv files found in {args.input_dir}")

    rows = []
    all_labels = []
    for attack_name, score_path in score_files:
        row, labels = calculate_far_for_file(attack_name, score_path)
        rows.append(row)
        all_labels.append(labels)

    combined_labels = pd.concat(all_labels, ignore_index=True)
    total = len(combined_labels)
    undetected = int((combined_labels == 0).sum())
    detected = int((combined_labels == 1).sum())
    far = undetected / total
    detection_rate = detected / total
    rows.append(
        {
            "Attack_Name": "ALL",
            "Total_Samples": total,
            "Detected_Samples": detected,
            "Undetected_Samples": undetected,
            "FAR": far,
            "FAR_Percent": far * 100,
            "Detection_Rate": detection_rate,
            "Detection_Rate_Percent": detection_rate * 100,
        }
    )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    display = result.copy()
    display["FAR"] = display["FAR"].map(lambda value: f"{value:.6f}")
    display["FAR_Percent"] = display["FAR_Percent"].map(lambda value: f"{value:.2f}%")
    display["Detection_Rate"] = display["Detection_Rate"].map(lambda value: f"{value:.6f}")
    display["Detection_Rate_Percent"] = display["Detection_Rate_Percent"].map(
        lambda value: f"{value:.2f}%"
    )
    print(display.to_string(index=False))
    print(f"\nSaved FAR results to: {args.output}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd


def time_score(claim_time, label_time):
    claim_time = float(claim_time)
    label_time = float(label_time)
    denominator = max(abs(claim_time), abs(label_time), 1e-9)
    relative_error = abs(claim_time - label_time) / denominator
    return max(0.0, min(1.0, 1.0 - relative_error))


def build_scores(claim_csv):
    claims = pd.read_csv(claim_csv)
    required = {"Claim_Path", "Claim_Time", "Label_Time"}
    missing = required - set(claims.columns)
    if missing:
        raise ValueError(f"Missing required columns in claim csv: {sorted(missing)}")

    results = []
    for _, row in claims.iterrows():
        results.append({
            "File_Name": row["Claim_Path"],
            "S_Time": round(time_score(row["Claim_Time"], row["Label_Time"]), 6),
        })

    return pd.DataFrame(results, columns=["File_Name", "S_Time"])


def main():
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    default_claim_csv = project_dir / "Claim" / "forged_trace_claims.csv"
    default_output = base_dir / "time_score.csv"

    parser = argparse.ArgumentParser(description="Calculate claim time credibility scores.")
    parser.add_argument("--claim-csv", type=Path, default=default_claim_csv)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    scores = build_scores(args.claim_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(scores)} scores to {args.output}")


if __name__ == "__main__":
    main()

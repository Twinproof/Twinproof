import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from Global_Anchor_Labeling import (
    extract_anchor_feature_row,
    load_global_rows,
    match_local_to_global,
)


PASSTHROUGH_COLUMNS = [
    "Signal_Source",
    "Label_Source",
    "Signal_Position",
    "Label_Position",
    "Claim_Location",
    "Label_Location",
    "Replay_Time_Diff",
    "Attack_Type",
]


def parse_position(value):
    if pd.isna(value):
        raise ValueError("empty position")
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"invalid position: {value}") from exc

    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"invalid position: {value}")
    start, end = int(parsed[0]), int(parsed[1])
    if end <= start:
        raise ValueError(f"invalid position range: {value}")
    return start, end


def signal_source_to_merged_path(signal_source, data_root):
    parts = str(signal_source).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Signal_Source: {signal_source}")

    folder = parts[0]
    stem = "/".join(parts[1:])
    base = data_root / folder
    candidates = []

    raw_path = base / stem
    candidates.append(raw_path)
    if raw_path.suffix.lower() != ".csv":
        candidates.append(raw_path.with_name(f"{raw_path.name}.csv"))
        if raw_path.name.endswith("_merged"):
            candidates.append(raw_path.with_suffix(".csv"))
        else:
            candidates.append(raw_path.with_name(f"{raw_path.name}_merged.csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Merged file not found for Signal_Source={signal_source}: {candidates[-1]}")


def build_location_local_rows(claims_file, data_root):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    required = ["Signal_Source", "Signal_Position", "Claim_Location"]
    missing = [column for column in required if column not in claims.columns]
    if missing:
        raise ValueError(f"{claims_file} missing columns: {missing}")

    rows = []
    data_cache = {}

    for claim_index, claim in claims.iterrows():
        signal_source = claim["Signal_Source"]
        signal_position = claim["Signal_Position"]
        try:
            merged_path = signal_source_to_merged_path(signal_source, data_root)
            start, end = parse_position(signal_position)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] claim row {claim_index}: {exc}")
            continue

        if merged_path not in data_cache:
            data_cache[merged_path] = pd.read_csv(merged_path, encoding="utf-8-sig")
        data = data_cache[merged_path]

        clipped_start = max(0, min(start, len(data)))
        clipped_end = max(clipped_start, min(end, len(data)))
        if clipped_end <= clipped_start:
            print(f"[skip] claim row {claim_index}: position outside file length {len(data)}")
            continue

        row = extract_anchor_feature_row(data, clipped_start, clipped_end, merged_path.name)
        row["Claim_Row_Index"] = claim_index
        for column in PASSTHROUGH_COLUMNS:
            if column in claims.columns:
                row[column] = claim.get(column)
        row["Source_Anchor_Info"] = str(signal_position)
        row["Anchor_Info"] = f"({clipped_start},{clipped_end})"
        rows.append(row)

    return pd.DataFrame(rows)


def build_location_global_anchor_table(
    claims_file,
    data_root,
    feature_dir,
    output_path,
    alpha,
    top_k,
    max_samples_per_anchor,
):
    global_features, global_labels = load_global_rows(feature_dir, max_samples_per_anchor)
    local_features = build_location_local_rows(claims_file, data_root)

    if local_features.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        local_features.to_csv(output_path, index=False, encoding="utf-8-sig")
        return local_features

    matches = match_local_to_global(global_features, global_labels, local_features, alpha, top_k)
    result = pd.concat([local_features.reset_index(drop=True), matches], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def default_output_for_claims(claims_file, project_root):
    stem = claims_file.stem.replace("_claims", "")
    return project_root / "Claim_Detection" / "anchor_with_global_id" / f"anchor_combined_{stem}_global.csv"


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build global anchor labels for location claims using Signal_Source and Signal_Position."
    )
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=project_root / "data" / "collectionData_02")
    parser.add_argument("--feature-dir", type=Path, default=project_root / "path_reconstruction" / "Anchor_feature_parking")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples-per-anchor", type=int, default=30)
    args = parser.parse_args()

    output_path = args.output or default_output_for_claims(args.claims, project_root)
    result = build_location_global_anchor_table(
        claims_file=args.claims,
        data_root=args.data_root,
        feature_dir=args.feature_dir,
        output_path=output_path,
        alpha=args.alpha,
        top_k=args.top_k,
        max_samples_per_anchor=args.max_samples_per_anchor,
    )

    print(f"[done] saved labeled location anchor table to: {output_path}")
    print(f"[done] rows: {len(result)}")
    if not result.empty and "Global_Anchor_ID" in result.columns:
        print(result["Global_Anchor_ID"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

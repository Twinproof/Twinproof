import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from S_Signal import (
    calculate_signal_scores,
    load_template_rows,
    preprocess_signal_features,
)


METADATA_COLUMNS = [
    "Claim_Row_Index",
    "File_Name",
    "Mid_Time",
    "Anchor_Info",
    "Source_Anchor_Info",
    "Signal_Source",
    "Signal_Position",
    "Claim_Location",
    "Label_Location",
    "Label_Source",
    "Label_Position",
    "Attack_Type",
]


def parse_region_set(value):
    if pd.isna(value):
        return set()

    text = str(value).strip()
    if not text:
        return set()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, str):
            return {parsed.strip()} if parsed.strip() else set()
        if isinstance(parsed, (list, tuple, set)):
            return {str(item).strip() for item in parsed if str(item).strip()}
    except (ValueError, SyntaxError):
        pass

    return {item.strip() for item in re.split(r"[,;，、\s]+", text) if item.strip()}


def load_anchor_region_mapping(connection_file):
    connection = pd.read_csv(connection_file, encoding="utf-8-sig")
    required = ["Cluster_Label", "Environment_Constraint"]
    missing = [column for column in required if column not in connection.columns]
    if missing:
        raise ValueError(f"{connection_file} missing columns: {missing}")

    mapping = {}
    for _, row in connection.iterrows():
        if pd.isna(row["Cluster_Label"]):
            continue
        mapping[int(row["Cluster_Label"])] = parse_region_set(row["Environment_Constraint"])
    return mapping


def topology_passes(row, anchor_regions):
    global_id_value = row.get("Global_Anchor_ID")
    if pd.isna(global_id_value):
        return False, set(), set()

    global_id = int(global_id_value)
    anchor_region = anchor_regions.get(global_id, set())
    claim_region = parse_region_set(row.get("Claim_Location"))
    return bool(anchor_region & claim_region), anchor_region, claim_region


def weighted_location_score(row, w_lte, w_meg, w_signal):
    return (
        w_lte * float(row["S_lte"])
        + w_meg * float(row["S_meg"])
        + w_signal * float(row["S_signal"])
    )


def validate_weights(w_lte, w_meg, w_signal):
    weight_sum = w_lte + w_meg + w_signal
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"Location weights must sum to 1.0, got {weight_sum}")


def calculate_location_scores(
    anchor_table,
    connection_file,
    feature_dir,
    output_path,
    meg_components,
    w_lte,
    w_meg,
    w_signal,
):
    validate_weights(w_lte, w_meg, w_signal)
    local_table = pd.read_csv(anchor_table, encoding="utf-8-sig")
    required = ["Global_Anchor_ID", "Claim_Location"]
    missing = [column for column in required if column not in local_table.columns]
    if missing:
        raise ValueError(f"{anchor_table} missing columns: {missing}")

    anchor_regions = load_anchor_region_mapping(connection_file)
    result = pd.DataFrame()
    for column in METADATA_COLUMNS:
        if column in local_table.columns:
            result[column] = local_table[column]

    result["Global_Anchor_ID"] = local_table["Global_Anchor_ID"]
    result["Anchor_Region"] = ""
    result["Claim_Region"] = ""
    result["Topology_Match"] = False
    result["S_topo"] = -1
    result["S_lte"] = -1.0
    result["S_meg"] = -1.0
    result["S_signal"] = -1.0
    result["S_location"] = -1.0

    passed_indices = []
    for index, row in local_table.iterrows():
        passed, anchor_region, claim_region = topology_passes(row, anchor_regions)
        result.at[index, "Anchor_Region"] = ",".join(sorted(anchor_region))
        result.at[index, "Claim_Region"] = ",".join(sorted(claim_region))
        result.at[index, "Topology_Match"] = passed
        result.at[index, "S_topo"] = 1 if passed else -1
        if passed:
            passed_indices.append(index)

    if passed_indices:
        passed_table = local_table.loc[passed_indices].reset_index(drop=True)
        template_features, template_labels = load_template_rows(feature_dir)
        feature_spaces = preprocess_signal_features(template_features, passed_table, meg_components)
        signal_scores = calculate_signal_scores(passed_table, template_features, template_labels, feature_spaces)

        for output_index, score_row in zip(passed_indices, signal_scores.to_dict("records")):
            result.at[output_index, "S_lte"] = score_row["S_lte"]
            result.at[output_index, "S_meg"] = score_row["S_meg"]
            result.at[output_index, "S_signal"] = score_row["S_signal"]
            result.at[output_index, "S_location"] = round(
                weighted_location_score(score_row, w_lte, w_meg, w_signal),
                6,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def default_output_for_anchor_table(anchor_table, project_root):
    name = anchor_table.stem
    name = name.replace("anchor_combined_", "").replace("_global", "")
    return project_root / "Claim_Detection" / "results" / "location" / name / "scores.csv"


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Calculate location attack scores with topology as a hard constraint before signal scoring."
    )
    parser.add_argument("--anchor-table", type=Path, required=True)
    parser.add_argument(
        "--connection-file",
        type=Path,
        default=project_root / "path_reconstruction" / "Anchor_connection.csv",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=project_root / "path_reconstruction" / "Anchor_feature_parking",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--meg-components", type=int, default=50)
    parser.add_argument("--w-lte", type=float, default=0.2)
    parser.add_argument("--w-meg", type=float, default=0.3)
    parser.add_argument("--w-signal", type=float, default=0.5)
    args = parser.parse_args()

    output_path = args.output or default_output_for_anchor_table(args.anchor_table, project_root)
    scores = calculate_location_scores(
        anchor_table=args.anchor_table,
        connection_file=args.connection_file,
        feature_dir=args.feature_dir,
        output_path=output_path,
        meg_components=args.meg_components,
        w_lte=args.w_lte,
        w_meg=args.w_meg,
        w_signal=args.w_signal,
    )

    print(f"[done] wrote {output_path}")
    print(f"[done] rows: {len(scores)}")
    print(scores[["S_topo", "S_lte", "S_meg", "S_signal", "S_location"]].head().to_string(index=False))


if __name__ == "__main__":
    main()

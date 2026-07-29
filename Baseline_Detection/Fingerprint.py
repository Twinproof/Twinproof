import argparse
import ast
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ATTACK_CONFIGS = [
    ("forged_trace", "Claim/forged_trace_claims.csv"),
    ("replay_trace", "Claim/replay_trace_claims.csv"),
    ("transplant_trace", "Claim/transplant_trace_claims.csv"),
]

RSSI_MISSING = -110.0
RSSI_SCALE = 18.0
MEG_SCALE = 25.0


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


def trace_key(value):
    return tuple(str(item) for item in parse_list(value))


def lcs_ratio(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    dp = np.zeros((len(left) + 1, len(right) + 1), dtype=int)
    for i, left_item in enumerate(left, start=1):
        for j, right_item in enumerate(right, start=1):
            if left_item == right_item:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return float(dp[-1, -1] / max(len(left), len(right), 1))


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
        key = (path.parent.name, path.name)
        by_folder_and_name[key] = path
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


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


@lru_cache(maxsize=None)
def extract_fingerprint(path_string):
    path = Path(path_string)
    df = pd.read_csv(path, encoding="utf-8-sig")

    rssi_by_cell = {}
    for index in range(1, 6):
        id_col = f"Cell_ID_{index}"
        rssi_col = f"Cell_RSSI_{index}"
        if id_col not in df.columns or rssi_col not in df.columns:
            continue

        cell_ids = pd.to_numeric(df[id_col], errors="coerce")
        rssi = pd.to_numeric(df[rssi_col], errors="coerce")
        valid = pd.DataFrame({"cell_id": cell_ids, "rssi": rssi}).dropna()
        if valid.empty:
            continue

        grouped = valid.groupby("cell_id")["rssi"].median()
        for cell_id, value in grouped.items():
            key = int(cell_id)
            rssi_by_cell.setdefault(key, []).append(float(value))

    rssi_fingerprint = {
        cell_id: float(np.mean(values))
        for cell_id, values in rssi_by_cell.items()
    }

    meg = numeric_series(df, "Meg")
    if meg.empty:
        meg_stats = np.zeros(4, dtype=float)
    else:
        diff = meg.diff().abs().dropna()
        meg_stats = np.array([
            float(meg.mean()),
            float(meg.std(ddof=0)),
            float(diff.mean()) if not diff.empty else 0.0,
            float(np.percentile(meg, 90) - np.percentile(meg, 10)),
        ])

    return {
        "path": str(path),
        "rssi": rssi_fingerprint,
        "meg": meg_stats,
    }


def fingerprint_distance(left, right):
    left_rssi = left["rssi"]
    right_rssi = right["rssi"]
    cell_ids = sorted(set(left_rssi) | set(right_rssi))

    if cell_ids:
        rssi_diffs = [
            left_rssi.get(cell_id, RSSI_MISSING) - right_rssi.get(cell_id, RSSI_MISSING)
            for cell_id in cell_ids
        ]
        rssi_distance = float(np.sqrt(np.mean(np.square(rssi_diffs))))
    else:
        rssi_distance = RSSI_SCALE * 2

    meg_distance = float(np.linalg.norm(left["meg"] - right["meg"]) / math.sqrt(len(left["meg"])))
    return 0.85 * (rssi_distance / RSSI_SCALE) + 0.15 * (meg_distance / MEG_SCALE)


def distance_to_score(distance):
    return float(1.0 / (1.0 + max(distance, 0.0)))


def build_reference_library(claim_files, data_root, by_folder_and_name, by_name):
    references = {}
    for claims_file in claim_files:
        if not claims_file.exists():
            continue
        claims = pd.read_csv(claims_file, encoding="utf-8-sig")
        for _, row in claims.iterrows():
            label_trace = row.get("Label_Trace", row.get("Claim_Trace"))
            key = trace_key(label_trace)
            if not key:
                continue

            ref_path_value = row.get("Label_Path")
            if pd.isna(ref_path_value) or not str(ref_path_value).strip():
                ref_path_value = row.get("Claim_Path")
            if pd.isna(ref_path_value) or not str(ref_path_value).strip():
                continue

            try:
                ref_path = resolve_merged_path(ref_path_value, data_root, by_folder_and_name, by_name)
                fingerprint = extract_fingerprint(str(ref_path))
            except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
                continue
            references.setdefault(key, []).append(fingerprint)

    if not references:
        raise ValueError("No reference fingerprints could be built from claim files.")
    return references


def best_reference_fingerprints(claim_trace, references):
    key = trace_key(claim_trace)
    if key in references:
        return references[key], key, 1.0

    best_key = max(references, key=lambda candidate: lcs_ratio(key, candidate))
    return references[best_key], best_key, lcs_ratio(key, best_key)


def submitted_path_for_row(row):
    attack_path = row.get("Attack_Path")
    if pd.notna(attack_path) and str(attack_path).strip():
        return attack_path
    return row.get("Claim_Path")


def score_claims(attack_name, claims_file, data_root, output_dir, references, by_folder_and_name, by_name):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    rows = []

    for _, row in claims.iterrows():
        claim_trace = row.get("Claim_Trace", row.get("Claim_Location", ""))
        submitted_path_value = submitted_path_for_row(row)

        try:
            submitted_path = resolve_merged_path(submitted_path_value, data_root, by_folder_and_name, by_name)
            submitted_fingerprint = extract_fingerprint(str(submitted_path))
            ref_fingerprints, ref_key, trace_similarity = best_reference_fingerprints(claim_trace, references)
            distances = [fingerprint_distance(submitted_fingerprint, ref) for ref in ref_fingerprints]
            distance = float(np.mean(sorted(distances)[: min(3, len(distances))]))
            if trace_similarity < 1.0:
                distance += (1.0 - trace_similarity)
            score = distance_to_score(distance)
            matched_file = submitted_path.name
        except Exception as exc:  # keep batch scoring robust and visible in the output
            distance = np.nan
            score = 0.0
            matched_file = ""
            ref_key = ()
            trace_similarity = 0.0
            print(f"[warn] {attack_name} {row.get('Claim_Path')}: {exc}")

        rows.append({
            "Attack_Name": attack_name,
            "Claim_Path": row.get("Claim_Path"),
            "Claim_Trace / Claim_Location": claim_trace,
            "Matched_File": matched_file,
            "Fingerprint_Score": round(float(score), 6),
            "Fingerprint_Distance": round(float(distance), 6) if pd.notna(distance) else np.nan,
        })

    result = pd.DataFrame(rows)
    attack_dir = output_dir / attack_name
    attack_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(attack_dir / "scores.csv", index=False, encoding="utf-8-sig")
    return result


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fingerprint verification baseline for trace attack claims.")
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--claim-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "fingerprint",
    )
    args = parser.parse_args()

    claim_files = [(name, args.claim_root / path) for name, path in ATTACK_CONFIGS]
    by_folder_and_name, by_name = build_merged_index(args.data_root)
    references = build_reference_library(
        [path for _, path in claim_files],
        args.data_root,
        by_folder_and_name,
        by_name,
    )

    all_results = []
    for attack_name, claims_file in claim_files:
        result = score_claims(
            attack_name,
            claims_file,
            args.data_root,
            args.output_dir,
            references,
            by_folder_and_name,
            by_name,
        )
        all_results.append(result)
        print(f"[done] {attack_name}: {len(result)} rows -> {args.output_dir / attack_name / 'scores.csv'}")

    combined = pd.concat(all_results, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "scores.csv", index=False, encoding="utf-8-sig")
    print(f"[done] combined: {len(combined)} rows -> {args.output_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()

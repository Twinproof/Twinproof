import argparse
import random
from pathlib import Path

import pandas as pd


SOURCE_MARKERS = [
    "route_004",
    "route_003",
]
OUTPUT_FOLDER_NAME = "route_007"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def path_for_log(path, base_dir):
    path = Path(path)
    if base_dir is None:
        return path.name
    try:
        return path.resolve().relative_to(Path(base_dir).resolve()).as_posix()
    except ValueError:
        return path.name


def claim_path_to_merged_path(claim_path, data_root):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")

    folder = parts[0]
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_merged"):
        file_name = f"{stem}.csv"
    else:
        file_name = f"{stem}_merged.csv"
    return data_root / folder / file_name


def collect_material_files(data_root, source_markers, rows):
    material_files = []
    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if not any(marker in folder.name for marker in source_markers):
            continue
        for file_path in sorted(folder.glob("*_merged.csv")):
            try:
                row_count = sum(1 for _ in open(file_path, "rb")) - 1
            except OSError:
                continue
            if row_count >= rows:
                material_files.append(file_path)

    if material_files:
        return material_files

    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for file_path in sorted(folder.glob("*_merged.csv")):
            row_count = sum(1 for _ in open(file_path, "rb")) - 1
            if row_count >= rows:
                material_files.append(file_path)

    if not material_files:
        raise FileNotFoundError(f"No *_merged.csv material file with at least {rows} rows found under {data_root}")
    return material_files


def sample_material_rows(material_files, target_columns, rows, rng):
    candidates = material_files[:]
    rng.shuffle(candidates)
    last_error = None

    for material_path in candidates:
        material_df = read_csv(material_path)
        if len(material_df) < rows:
            continue
        if list(material_df.columns) != list(target_columns):
            last_error = f"Column mismatch: {material_path}"
            continue

        start = rng.randint(0, len(material_df) - rows)
        end = start + rows
        return material_path, start, end, material_df.iloc[start:end].copy()

    raise ValueError(last_error or "No material file with matching columns was found")


def build_replay_trace_last(claims_file, data_root, output_dir, rows, seed, log_base_dir=None):
    rng = random.Random(seed)
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    material_files = collect_material_files(data_root, SOURCE_MARKERS, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for claim_index, claim_path in enumerate(claims["Claim_Path"].astype(str).tolist()):
        original_path = claim_path_to_merged_path(claim_path, data_root)
        if not original_path.exists():
            print(
                f"[skip] original file not found for claim row {claim_index}: "
                f"{path_for_log(original_path, log_base_dir)}"
            )
            continue

        original_df = read_csv(original_path)
        material_path, start, end, sampled_df = sample_material_rows(
            material_files=material_files,
            target_columns=original_df.columns,
            rows=rows,
            rng=rng,
        )
        attacked_df = pd.concat([original_df, sampled_df], ignore_index=True)
        output_path = output_dir / f"claim_{claim_index:03d}_{original_path.name}"
        attacked_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        reports.append({
            "Claim_Index": claim_index,
            "Claim_Path": claim_path,
            "Output_File": path_for_log(output_path, log_base_dir),
            "Original_File": path_for_log(original_path, log_base_dir),
            "Material_File": path_for_log(material_path, log_base_dir),
            "Material_Start_Row": start,
            "Material_End_Row_Exclusive": end,
            "Original_Rows": len(original_df),
            "Appended_Rows": rows,
            "Output_Rows": len(attacked_df),
        })
        print(
            f"[append] claim {claim_index:03d} {output_path.name}: "
            f"{len(original_df)} -> {len(attacked_df)} from {material_path.name} rows {start}:{end}"
        )

    return pd.DataFrame(reports)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build replay trace *_last merged files by appending 16 seconds of data.")
    parser.add_argument("--claims", type=Path, default=project_root / "Claim" / "replay_trace_claims.csv")
    parser.add_argument("--data-root", type=Path, default=project_root / "data" / "collectionData_02")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "collectionData_02" / OUTPUT_FOLDER_NAME,
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "replay_trace_last_append_log.csv",
    )
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true", help="Remove existing *_merged.csv files in the output folder first.")
    args = parser.parse_args()

    if args.clear_output and args.output_dir.exists():
        for old_file in args.output_dir.glob("*_merged.csv"):
            old_file.unlink()

    reports = build_replay_trace_last(
        claims_file=args.claims,
        data_root=args.data_root,
        output_dir=args.output_dir,
        rows=args.rows,
        seed=args.seed,
        log_base_dir=project_root,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    reports.to_csv(args.log, index=False, encoding="utf-8-sig")
    print(
        f"[done] wrote {len(reports)} attacked files to "
        f"{path_for_log(args.output_dir, project_root)}"
    )
    print(f"[done] wrote log to {path_for_log(args.log, project_root)}")


if __name__ == "__main__":
    main()

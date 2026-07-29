import argparse
import random
from pathlib import Path

import pandas as pd


def find_folder(root, marker):
    matches = [path for path in root.iterdir() if path.is_dir() and marker in path.name]
    if not matches:
        raise FileNotFoundError(f"No folder containing {marker!r} found under {root}")
    if len(matches) > 1:
        print(f"[warn] multiple folders containing {marker!r}, using {matches[0]}")
    return matches[0]


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path), "default"


def file_kind(path):
    name = path.name.lower()
    if name.startswith("sensor"):
        return "sensor"
    if name.startswith("signal"):
        return "signal"
    return None


def sample_continuous_rows(source_df, rows, rng):
    if len(source_df) < rows:
        raise ValueError(f"Source file has only {len(source_df)} rows, fewer than requested {rows}")
    start = rng.randint(0, len(source_df) - rows)
    end = start + rows
    return source_df.iloc[start:end].copy(), start, end


def rows_for_kind(kind, sensor_rows, signal_rows):
    if kind == "sensor":
        return sensor_rows
    if kind == "signal":
        return signal_rows
    raise ValueError(f"Unsupported file kind: {kind}")


def append_rows_to_targets(source_dir, target_dir, sensor_rows, signal_rows, seed):
    rng = random.Random(seed)
    source_files = sorted(source_dir.glob("*.csv"))
    target_files = sorted(target_dir.glob("*.csv"))

    source_by_kind = {"sensor": [], "signal": []}
    for source_file in source_files:
        kind = file_kind(source_file)
        if kind in source_by_kind:
            source_by_kind[kind].append(source_file)

    source_cache = {}
    for kind, files in source_by_kind.items():
        required_rows = rows_for_kind(kind, sensor_rows, signal_rows)
        for source_file in files:
            source_df, _ = read_csv(source_file)
            source_cache[source_file] = source_df
            if len(source_df) < required_rows:
                raise ValueError(
                    f"Source {kind} file {source_file.name} has only {len(source_df)} rows, "
                    f"fewer than requested {required_rows}"
                )

    reports = []
    for target_file in target_files:
        kind = file_kind(target_file)
        if kind is None:
            print(f"[skip] unsupported target file name: {target_file.name}")
            continue
        if not source_by_kind[kind]:
            raise FileNotFoundError(f"No {kind} source csv found in {source_dir}")

        source_file = rng.choice(source_by_kind[kind])
        source_df = source_cache[source_file]
        target_df, target_encoding = read_csv(target_file)
        required_rows = rows_for_kind(kind, sensor_rows, signal_rows)

        if list(source_df.columns) != list(target_df.columns):
            raise ValueError(
                f"Column mismatch between source {source_file.name} and target {target_file.name}"
            )

        sampled_df, start, end = sample_continuous_rows(source_df, required_rows, rng)
        combined_df = pd.concat([target_df, sampled_df], ignore_index=True)
        combined_df.to_csv(target_file, index=False, encoding=target_encoding if target_encoding != "default" else "utf-8-sig")

        reports.append({
            "target_file": target_file.name,
            "source_file": source_file.name,
            "source_start_row": start,
            "source_end_row_exclusive": end,
            "original_target_rows": len(target_df),
            "new_target_rows": len(combined_df),
        })

    return reports


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Append random continuous 16-second rows from route_003 to route_004 CSV files."
    )
    parser.add_argument("--root", type=Path, default=project_root / "data" / "collectionData")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--source-marker", default="route_003")
    parser.add_argument("--target-marker", default="route_004")
    parser.add_argument("--sensor-rows", type=int, default=800, help="16 seconds at 50 Hz.")
    parser.add_argument("--signal-rows", type=int, default=8, help="16 seconds at 0.5 Hz.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    source_root = args.source_root or args.root
    target_root = args.target_root or args.root
    source_dir = find_folder(source_root, args.source_marker)
    target_dir = find_folder(target_root, args.target_marker)
    reports = append_rows_to_targets(source_dir, target_dir, args.sensor_rows, args.signal_rows, args.seed)

    print(f"[done] source: {source_dir}")
    print(f"[done] target: {target_dir}")
    for report in reports:
        print(
            "[append] "
            f"{report['target_file']} <- {report['source_file']} "
            f"rows {report['source_start_row']}:{report['source_end_row_exclusive']} "
            f"({report['original_target_rows']} -> {report['new_target_rows']})"
        )


if __name__ == "__main__":
    main()

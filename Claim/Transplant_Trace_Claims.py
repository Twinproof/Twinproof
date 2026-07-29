import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


LABEL_TRACE = ["e4", "d4", "e4"]
CLAIM_TRACE = ["e4", "e3", "d3", "d4", "e4"]
ATTACK_TYPE = "transplant_attack"
FIELDNAMES = [
    "Claim_Path",
    "Label_Path",
    "Claim_Time",
    "Claim_Trace",
    "Label_Time",
    "Label_Trace",
    "Attack_Type",
]


def parse_time(value):
    return datetime.strptime(value.strip(), "%H:%M:%S:%f")


def duration_seconds(sensor_path):
    with sensor_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        first_time = None
        last_time = None

        for row in reader:
            current_time = row.get("Time")
            if not current_time:
                continue
            if first_time is None:
                first_time = current_time
            last_time = current_time

    if first_time is None or last_time is None:
        raise ValueError(f"No valid Time values found in {sensor_path}")

    start = parse_time(first_time)
    end = parse_time(last_time)
    if end < start:
        end += timedelta(days=1)

    return round((end - start).total_seconds(), 3)


def claim_path_from_sensor(sensor_path, data_dir):
    stem = sensor_path.stem
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    return f"{data_dir.name}/{stem}"


def parse_trace(value):
    trace = [item.strip() for item in value.split(",") if item.strip()]
    if not trace:
        raise ValueError("trace cannot be empty")
    return trace


def build_rows(current_data_dir, transplant_data_dir, claim_trace, label_trace):
    current_sensor_files = sorted(current_data_dir.glob("sensor*.csv"))
    transplant_sensor_files = sorted(transplant_data_dir.glob("sensor*.csv"))

    if not current_sensor_files:
        raise ValueError(f"No sensor csv files found in current data dir: {current_data_dir}")
    if not transplant_sensor_files:
        raise ValueError(f"No sensor csv files found in transplant data dir: {transplant_data_dir}")

    rows = []
    for current_sensor_path in current_sensor_files:
        transplant_sensor_path = random.choice(transplant_sensor_files)

        rows.append({
            "Claim_Path": claim_path_from_sensor(transplant_sensor_path, transplant_data_dir),
            "Label_Path": claim_path_from_sensor(current_sensor_path, current_data_dir),
            "Claim_Time": duration_seconds(transplant_sensor_path),
            "Claim_Trace": json.dumps(claim_trace, ensure_ascii=False),
            "Label_Time": duration_seconds(current_sensor_path),
            "Label_Trace": json.dumps(label_trace, ensure_ascii=False),
            "Attack_Type": ATTACK_TYPE,
        })

    return rows


def append_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    claim_dir = Path(__file__).resolve().parent
    collection_dir = claim_dir.parent / "data" / "collectionData"
    default_current_data_dir = collection_dir / "route_006"
    default_transplant_data_dir = collection_dir / "route_007"
    default_output = claim_dir / "transplant_trace_claims.csv"

    parser = argparse.ArgumentParser(description="Generate transplanted trace attack samples.")
    parser.add_argument("--current-data-dir", type=Path, default=default_current_data_dir)
    parser.add_argument("--transplant-data-dir", type=Path, default=default_transplant_data_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--claim-trace", default=",".join(CLAIM_TRACE))
    parser.add_argument("--label-trace", default=",".join(LABEL_TRACE))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.current_data_dir.exists():
        raise FileNotFoundError(f"Current data directory not found: {args.current_data_dir}")
    if not args.transplant_data_dir.exists():
        raise FileNotFoundError(f"Transplant data directory not found: {args.transplant_data_dir}")

    claim_trace = parse_trace(args.claim_trace)
    label_trace = parse_trace(args.label_trace)
    rows = build_rows(args.current_data_dir, args.transplant_data_dir, claim_trace, label_trace)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} transplanted trace claims to {args.output}")
    print(f"Claim_Path sampled from {args.transplant_data_dir}")
    print(f"Label_Path sampled from {args.current_data_dir}")


if __name__ == "__main__":
    main()

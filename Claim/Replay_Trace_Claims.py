import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


TRUE_TRACE = ["e4", "e3", "e4", "e3", "d3", "d4", "e4", "d4"]
ATTACK_TYPE = "replay_attack"
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


def build_rows(current_data_dir, replay_data_dir):
    current_sensor_files = sorted(current_data_dir.glob("sensor*.csv"))
    replay_sensor_files = sorted(replay_data_dir.glob("sensor*.csv"))

    if not current_sensor_files:
        raise ValueError(f"No sensor csv files found in current data dir: {current_data_dir}")
    if not replay_sensor_files:
        raise ValueError(f"No sensor csv files found in replay data dir: {replay_data_dir}")

    rows = []
    for current_sensor_path in current_sensor_files:
        replay_sensor_path = random.choice(replay_sensor_files)
        label_trace = list(TRUE_TRACE)
        claim_trace = list(TRUE_TRACE)

        rows.append({
            "Claim_Path": claim_path_from_sensor(replay_sensor_path, replay_data_dir),
            "Label_Path": claim_path_from_sensor(current_sensor_path, current_data_dir),
            "Claim_Time": duration_seconds(replay_sensor_path),
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
    default_current_data_dir = collection_dir / "route_004"
    default_replay_data_dir = collection_dir / "route_006"
    default_output = claim_dir / "replay_trace_claims.csv"

    parser = argparse.ArgumentParser(description="Generate replay attack trace claim samples.")
    parser.add_argument("--current-data-dir", type=Path, default=default_current_data_dir)
    parser.add_argument("--replay-data-dir", type=Path, default=default_replay_data_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.current_data_dir.exists():
        raise FileNotFoundError(f"Current data directory not found: {args.current_data_dir}")
    if not args.replay_data_dir.exists():
        raise FileNotFoundError(f"Replay data directory not found: {args.replay_data_dir}")

    rows = build_rows(args.current_data_dir, args.replay_data_dir)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} replay trace claims to {args.output}")
    print(f"Claim_Path sampled from {args.replay_data_dir}")
    print(f"Label_Path sampled from {args.current_data_dir}")


if __name__ == "__main__":
    main()

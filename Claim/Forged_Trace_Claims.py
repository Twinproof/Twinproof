import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


TRUE_TRACE = ["e4", "e3", "e4", "e3","d3", "d4", "e4", "d4"]
NODE_POOL = [f"{row}{col}" for row in "abcdef" for col in range(1, 7)]
DEFAULT_TIME_ATTACK_RATE = 0.50
ATTACK_TYPE = "forged_claim"
FIELDNAMES = ["Claim_Path", "Claim_Time", "Claim_Trace", "Label_Time", "Label_Trace", "Attack_Type"]


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


def mutate_time(label_time):
    # Keep the generated time plausible while making it different from the label.
    offset = random.uniform(-0.25, 0.25) * label_time
    if abs(offset) < 1:
        offset = 1 if random.random() < 0.5 else -1
    return round(max(0.001, label_time + offset), 3)


def mutate_trace(label_trace):
    trace = list(label_trace)
    mutation_count = random.randint(1, min(3, len(trace)))
    indexes = random.sample(range(len(trace)), mutation_count)

    for index in indexes:
        choices = [node for node in NODE_POOL if node != trace[index]]
        trace[index] = random.choice(choices)

    if trace == label_trace:
        return mutate_trace(label_trace)
    return trace


def build_rows(data_dir, time_attack_rate):
    sensor_files = sorted(data_dir.glob("sensor*.csv"))
    time_attack_count = int(len(sensor_files) * time_attack_rate)
    time_attack_indexes = (
        set(random.sample(range(len(sensor_files)), time_attack_count))
        if time_attack_count
        else set()
    )

    rows = []
    for index, sensor_path in enumerate(sensor_files):
        label_time = duration_seconds(sensor_path)
        label_trace = list(TRUE_TRACE)
        claim_time = label_time
        claim_trace = mutate_trace(label_trace)

        if index in time_attack_indexes:
            claim_time = mutate_time(label_time)

        rows.append({
            "Claim_Path": claim_path_from_sensor(sensor_path, data_dir),
            "Claim_Time": claim_time,
            "Claim_Trace": json.dumps(claim_trace, ensure_ascii=False),
            "Label_Time": label_time,
            "Label_Trace": json.dumps(label_trace, ensure_ascii=False),
            "Attack_Type": ATTACK_TYPE,
        })

    return rows, time_attack_count


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
    default_data_dir = claim_dir.parent / "data" / "collectionData" / "route_004"
    default_output = claim_dir / "forged_trace_claims.csv"

    parser = argparse.ArgumentParser(description="Generate forged trace claim attack samples.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--time-attack-rate", type=float, default=DEFAULT_TIME_ATTACK_RATE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not 0 <= args.time_attack_rate <= 1:
        raise ValueError("time-attack-rate must be between 0 and 1")

    rows, time_attack_count = build_rows(args.data_dir, args.time_attack_rate)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} forged trace claims to {args.output}")
    print(f"Mutated {len(rows)} Claim_Trace values")
    print(f"Mutated {time_attack_count} Claim_Time values")


if __name__ == "__main__":
    main()

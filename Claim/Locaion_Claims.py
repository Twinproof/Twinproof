import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path


DEFAULT_ERROR_RATE = 0.30
DATASET_NAME = "route_004"
NODE_POOL = [f"{row}{col}" for row in "abcdef" for col in range(1, 7)]
FIELDNAMES = ["Signal_Source", "Signal_Position", "Claim_Location", "Label_Location"]


def mid_time_to_seconds(value):
    parsed = datetime.strptime(value.strip(), "%H:%M:%S").time()
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def location_from_mid_time(mid_time):
    seconds = mid_time_to_seconds(mid_time)

    if 0 <= seconds < 50:
        return ["e4"]
    if 50 <= seconds < 100:
        return ["d3", "d4"]
    if 100 <= seconds < 150:
        return ["d3", "c4"]
    if seconds >= 150:
        return ["d4"]

    raise ValueError(f"Mid_Time out of supported range: {mid_time}")


def mutate_location(label_location):
    claim_location = list(label_location)
    mutation_count = random.randint(1, len(claim_location))
    indexes = random.sample(range(len(claim_location)), mutation_count)

    for index in indexes:
        choices = [node for node in NODE_POOL if node != claim_location[index]]
        claim_location[index] = random.choice(choices)

    if claim_location == label_location:
        return mutate_location(label_location)
    return claim_location


def signal_source_from_file_name(file_name):
    source_name = Path(file_name).stem
    if source_name.endswith("_merged"):
        source_name = source_name[:-len("_merged")]
    return f"{DATASET_NAME}/{source_name}"


def build_rows(input_path, error_rate):
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"File_Name", "Mid_Time", "Anchor_Info"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        source_rows = list(reader)

    error_count = int(len(source_rows) * error_rate)
    error_indexes = set(random.sample(range(len(source_rows)), error_count)) if error_count else set()

    rows = []
    for index, source_row in enumerate(source_rows):
        label_location = location_from_mid_time(source_row["Mid_Time"])
        claim_location = mutate_location(label_location) if index in error_indexes else list(label_location)

        rows.append({
            "Signal_Source": signal_source_from_file_name(source_row["File_Name"]),
            "Signal_Position": source_row["Anchor_Info"],
            "Claim_Location": json.dumps(claim_location, ensure_ascii=False),
            "Label_Location": json.dumps(label_location, ensure_ascii=False),
        })

    return rows, error_count


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
    default_input = claim_dir.parent / "Find_Anchor" / "anchor" / "anchor_combined_route_004.csv"
    default_output = claim_dir / "location_claims.csv"

    parser = argparse.ArgumentParser(description="Generate location claims from anchor csv rows.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--error-rate", type=float, default=DEFAULT_ERROR_RATE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not 0 <= args.error_rate <= DEFAULT_ERROR_RATE:
        raise ValueError(f"error-rate must be between 0 and {DEFAULT_ERROR_RATE}")

    rows, error_count = build_rows(args.input, args.error_rate)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} location claims to {args.output}")
    print(f"Generated {error_count} wrong Claim_Location values")


if __name__ == "__main__":
    main()

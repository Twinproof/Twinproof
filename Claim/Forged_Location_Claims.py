import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path


DATASET_NAME = "route_004"
ATTACK_TYPE = "forged_location"
NODE_POOL = [f"{row}{col}" for row in "abcdef" for col in range(1, 7)]
FIELDNAMES = ["Signal_Source", "Signal_Position", "Claim_Location", "Label_Location", "Attack_Type"]


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


def build_rows(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"File_Name", "Mid_Time", "Anchor_Info"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        source_rows = list(reader)

    rows = []
    for source_row in source_rows:
        label_location = location_from_mid_time(source_row["Mid_Time"])
        claim_location = mutate_location(label_location)

        rows.append({
            "Signal_Source": signal_source_from_file_name(source_row["File_Name"]),
            "Signal_Position": source_row["Anchor_Info"],
            "Claim_Location": json.dumps(claim_location, ensure_ascii=False),
            "Label_Location": json.dumps(label_location, ensure_ascii=False),
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
    default_input = claim_dir.parent / "Find_Anchor" / "anchor" / "anchor_combined_route_004.csv"
    default_output = claim_dir / "forged_location_claims.csv"

    parser = argparse.ArgumentParser(description="Generate forged location claim attack samples.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = build_rows(args.input)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} forged location claims to {args.output}")
    print(f"Mutated {len(rows)} Claim_Location values")


if __name__ == "__main__":
    main()

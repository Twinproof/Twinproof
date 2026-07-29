import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


CURRENT_DATASET_NAME = "route_004"
REPLAY_DATASET_NAME = "route_003"
ATTACK_TYPE = "replay_location"
FIELDNAMES = [
    "Signal_Source",
    "Label_Source",
    "Signal_Position",
    "Label_Position",
    "Claim_Location",
    "Label_Location",
    "Replay_Time_Diff",
    "Attack_Type",
]


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


def source_from_file_name(file_name, dataset_name):
    source_name = Path(file_name).stem
    if source_name.endswith("_merged"):
        source_name = source_name[:-len("_merged")]
    return f"{dataset_name}/{source_name}"


def read_anchor_rows(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"File_Name", "Mid_Time", "Anchor_Info"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns in {input_path}: {sorted(missing_columns)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"No anchor rows found in {input_path}")

    for row in rows:
        row["_Mid_Time_Seconds"] = mid_time_to_seconds(row["Mid_Time"])

    return rows


def find_nearest_replay_row(current_row, replay_rows):
    current_seconds = current_row["_Mid_Time_Seconds"]
    return min(replay_rows, key=lambda row: abs(row["_Mid_Time_Seconds"] - current_seconds))


def build_rows(current_input, replay_input):
    current_rows = read_anchor_rows(current_input)
    replay_rows = read_anchor_rows(replay_input)

    rows = []
    for current_row in current_rows:
        replay_row = find_nearest_replay_row(current_row, replay_rows)
        label_location = location_from_mid_time(current_row["Mid_Time"])
        replay_time_diff = abs(replay_row["_Mid_Time_Seconds"] - current_row["_Mid_Time_Seconds"])

        rows.append({
            "Signal_Source": source_from_file_name(replay_row["File_Name"], REPLAY_DATASET_NAME),
            "Label_Source": source_from_file_name(current_row["File_Name"], CURRENT_DATASET_NAME),
            "Signal_Position": replay_row["Anchor_Info"],
            "Label_Position": current_row["Anchor_Info"],
            "Claim_Location": json.dumps(label_location, ensure_ascii=False),
            "Label_Location": json.dumps(label_location, ensure_ascii=False),
            "Replay_Time_Diff": replay_time_diff,
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
    anchor_dir = claim_dir.parent / "Find_Anchor" / "anchor"
    default_current_input = anchor_dir / "anchor_combined_route_004.csv"
    default_replay_input = anchor_dir / "anchor_combined_route_003.csv"
    default_output = claim_dir / "replay_location_claims.csv"

    parser = argparse.ArgumentParser(description="Generate replay attack location claim samples.")
    parser.add_argument("--current-input", type=Path, default=default_current_input)
    parser.add_argument("--replay-input", type=Path, default=default_replay_input)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    if not args.current_input.exists():
        raise FileNotFoundError(f"Current anchor file not found: {args.current_input}")
    if not args.replay_input.exists():
        raise FileNotFoundError(f"Replay anchor file not found: {args.replay_input}")

    rows = build_rows(args.current_input, args.replay_input)
    append_rows(args.output, rows)

    print(f"Appended {len(rows)} replay location claims to {args.output}")
    print(f"Signal_Source matched from {args.replay_input}")
    print(f"Label_Source matched from {args.current_input}")


if __name__ == "__main__":
    main()

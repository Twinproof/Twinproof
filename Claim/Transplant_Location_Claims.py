import argparse
import ast
import csv
import random
from functools import lru_cache
from pathlib import Path


SEGMENT_LENGTH = 200
DEFAULT_SEED = 20260709
ATTACK_TYPE = "transplant_location"
OUTPUT_FOLDER_NAME = "route_003_transplant"
OUTPUT_STEM_PREFIX = "route_003_transplant"

REQUIRED_COLUMNS = {
    "Signal_Source",
    "Label_Source",
    "Signal_Position",
    "Label_Position",
}

MANIFEST_FIELDS = [
    "Sample_ID",
    "Output_Source",
    "Output_File",
    "Output_Position",
    "Original_Signal_Source",
    "Sampled_Signal_Position",
    "Original_Label_Source",
    "Sampled_Label_Position",
    "Random_Seed",
]


def parse_position(value, field_name):
    try:
        parsed = ast.literal_eval(str(value).strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc

    if (
        not isinstance(parsed, (tuple, list))
        or len(parsed) != 2
        or not all(isinstance(item, int) for item in parsed)
    ):
        raise ValueError(f"{field_name} must be an integer pair: {value!r}")

    start, end = parsed
    if start < 0 or end <= start:
        raise ValueError(f"Invalid {field_name} interval: {value!r}")
    if end - start < SEGMENT_LENGTH:
        raise ValueError(
            f"{field_name} interval is shorter than {SEGMENT_LENGTH} rows: {value!r}"
        )
    return start, end


def source_to_path(data_root, source):
    normalized = str(source).strip().replace("\\", "/")
    relative_path = Path(*normalized.split("/"))

    if relative_path.suffix.lower() == ".csv":
        candidates = [data_root / relative_path]
    else:
        candidates = [
            data_root / relative_path.with_name(f"{relative_path.name}_merged.csv"),
            data_root / relative_path.with_suffix(".csv"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve source {source!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


@lru_cache(maxsize=None)
def read_source(path_string):
    path = Path(path_string)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Source file has no header: {path}")
        return tuple(reader.fieldnames), tuple(reader)


def sample_segment(data_root, source, position, rng, field_name):
    source_path = source_to_path(data_root, source)
    fieldnames, rows = read_source(str(source_path.resolve()))
    range_start, range_end = parse_position(position, field_name)

    if range_end > len(rows):
        raise ValueError(
            f"{field_name} {position!r} exceeds {source_path} "
            f"with {len(rows)} data rows"
        )

    sampled_start = rng.randint(range_start, range_end - SEGMENT_LENGTH)
    sampled_end = sampled_start + SEGMENT_LENGTH
    return fieldnames, list(rows[sampled_start:sampled_end]), (sampled_start, sampled_end)


def write_sample(output_path, fieldnames, rows):
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(output_path, fieldnames, rows):
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_samples(input_claims, data_root, output_dir, output_claims, seed):
    with input_claims.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        claim_fields = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(claim_fields)
        if missing:
            raise ValueError(
                f"Missing required columns in {input_claims}: {sorted(missing)}"
            )
        source_claims = list(reader)

    if not source_claims:
        raise ValueError(f"No claims found in {input_claims}")
    if output_claims.exists():
        raise FileExistsError(f"Output claims already exists: {output_claims}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_claims.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    transplant_claims = []
    manifest_rows = []

    for index, claim in enumerate(source_claims, start=1):
        signal_fields, signal_rows, signal_window = sample_segment(
            data_root,
            claim["Signal_Source"],
            claim["Signal_Position"],
            rng,
            "Signal_Position",
        )
        label_fields, label_rows, label_window = sample_segment(
            data_root,
            claim["Label_Source"],
            claim["Label_Position"],
            rng,
            "Label_Position",
        )

        if signal_fields != label_fields:
            raise ValueError(
                f"Column mismatch between {claim['Signal_Source']} "
                f"and {claim['Label_Source']}"
            )

        sample_id = f"{index:04d}"
        output_stem = f"{OUTPUT_STEM_PREFIX}_{sample_id}"
        output_file = output_dir / f"{output_stem}_merged.csv"
        output_source = f"{OUTPUT_FOLDER_NAME}/{output_stem}"
        combined_rows = signal_rows + label_rows

        if len(combined_rows) != SEGMENT_LENGTH * 2:
            raise RuntimeError(
                f"Expected {SEGMENT_LENGTH * 2} rows for {output_stem}, "
                f"got {len(combined_rows)}"
            )

        write_sample(output_file, signal_fields, combined_rows)

        new_claim = dict(claim)
        new_claim["Signal_Source"] = output_source
        new_claim["Signal_Position"] = f"(0,{SEGMENT_LENGTH * 2})"
        if "Attack_Type" in new_claim:
            new_claim["Attack_Type"] = ATTACK_TYPE
        transplant_claims.append(new_claim)

        manifest_rows.append(
            {
                "Sample_ID": sample_id,
                "Output_Source": output_source,
                "Output_File": output_file.name,
                "Output_Position": f"(0,{SEGMENT_LENGTH * 2})",
                "Original_Signal_Source": claim["Signal_Source"],
                "Sampled_Signal_Position": f"({signal_window[0]},{signal_window[1]})",
                "Original_Label_Source": claim["Label_Source"],
                "Sampled_Label_Position": f"({label_window[0]},{label_window[1]})",
                "Random_Seed": seed,
            }
        )

    write_csv(output_claims, claim_fields, transplant_claims)
    manifest_path = output_dir / "transplant_location_manifest.csv"
    write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    return len(transplant_claims), manifest_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build 400-row location trajectory transplantation samples."
    )
    parser.add_argument(
        "--input-claims",
        type=Path,
        default=project_root / "Claim" / "replay_location_claims.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "collectionData_02",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "data"
        / "collectionData_02"
        / OUTPUT_FOLDER_NAME,
    )
    parser.add_argument(
        "--output-claims",
        type=Path,
        default=project_root / "Claim" / "transplant_location_claims.csv",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.input_claims.exists():
        raise FileNotFoundError(f"Claims file not found: {args.input_claims}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    count, manifest_path = build_samples(
        args.input_claims,
        args.data_root,
        args.output_dir,
        args.output_claims,
        args.seed,
    )
    print(f"Generated {count} transplant location samples.")
    print(f"Samples: {args.output_dir}")
    print(f"Claims: {args.output_claims}")
    print(f"Manifest: {manifest_path}")
    print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()

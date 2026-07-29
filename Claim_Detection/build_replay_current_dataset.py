import argparse
import random
from pathlib import Path

import pandas as pd


SENSOR_APPEND_ROWS = 800
SIGNAL_APPEND_ROWS = 8


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def claim_path_to_source_paths(claim_path, data_root):
    parts = str(claim_path).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid Claim_Path: {claim_path}")
    folder = parts[0]
    stem = parts[-1]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    if stem.startswith("signal_"):
        stem = stem[len("signal_"):]
    source_dir = data_root / folder
    return source_dir / f"sensor_{stem}.csv", source_dir / f"signal_{stem}.csv", stem


def paired_signal_path(sensor_path):
    return sensor_path.with_name(sensor_path.name.replace("sensor_", "signal_", 1))


def collect_material_pairs(data_root, preferred_keywords, excluded_dirs):
    preferred = []
    fallback = []

    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if folder.name in excluded_dirs:
            continue
        bucket = preferred if any(keyword in folder.name for keyword in preferred_keywords) else fallback
        for sensor_path in sorted(folder.glob("sensor_*.csv")):
            signal_path = paired_signal_path(sensor_path)
            if signal_path.exists():
                bucket.append((sensor_path, signal_path))

    return preferred, fallback


def choose_material_segment(preferred_pairs, fallback_pairs, rng):
    candidates = preferred_pairs[:]
    rng.shuffle(candidates)
    fallback_candidates = fallback_pairs[:]
    rng.shuffle(fallback_candidates)

    for pool in (candidates, fallback_candidates):
        for sensor_path, signal_path in pool:
            sensor_df = read_csv(sensor_path)
            signal_df = read_csv(signal_path)
            if len(sensor_df) < SENSOR_APPEND_ROWS or len(signal_df) < SIGNAL_APPEND_ROWS:
                continue

            max_sensor_start = len(sensor_df) - SENSOR_APPEND_ROWS
            sensor_start = rng.randint(0, max_sensor_start)
            signal_start = min(sensor_start // 100, len(signal_df) - SIGNAL_APPEND_ROWS)
            signal_start = max(0, signal_start)

            return {
                "sensor_path": sensor_path,
                "signal_path": signal_path,
                "sensor_df": sensor_df,
                "signal_df": signal_df,
                "sensor_start": sensor_start,
                "sensor_end": sensor_start + SENSOR_APPEND_ROWS,
                "signal_start": signal_start,
                "signal_end": signal_start + SIGNAL_APPEND_ROWS,
            }

    raise ValueError("No material sensor/signal pair has enough rows.")


def output_paths(output_dir, claim_index, stem):
    safe_stem = Path(stem).stem
    base = f"claim_{claim_index:03d}_{safe_stem}"
    return output_dir / f"sensor_{base}.csv", output_dir / f"signal_{base}.csv"


def build_dataset(claims_file, data_root, output_dir, seed):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    if "Claim_Path" not in claims.columns:
        raise ValueError(f"{claims_file} missing Claim_Path column")

    output_dir.mkdir(parents=False, exist_ok=False)
    rng = random.Random(seed)
    preferred_pairs, fallback_pairs = collect_material_pairs(
        data_root=data_root,
        preferred_keywords=["route_004", "route_003"],
        excluded_dirs={"route_006", "route_007"},
    )
    if not preferred_pairs and not fallback_pairs:
        raise FileNotFoundError(f"No material sensor/signal pairs found under {data_root}")

    manifest_rows = []
    for claim_index, row in claims.iterrows():
        base_sensor_path, base_signal_path, stem = claim_path_to_source_paths(row["Claim_Path"], data_root)
        if not base_sensor_path.exists() or not base_signal_path.exists():
            raise FileNotFoundError(f"Missing base sensor/signal for {row['Claim_Path']}")

        base_sensor = read_csv(base_sensor_path)
        base_signal = read_csv(base_signal_path)
        material = choose_material_segment(preferred_pairs, fallback_pairs, rng)

        append_sensor = material["sensor_df"].iloc[material["sensor_start"]:material["sensor_end"]]
        append_signal = material["signal_df"].iloc[material["signal_start"]:material["signal_end"]]
        attack_sensor = pd.concat([base_sensor, append_sensor], ignore_index=True)
        attack_signal = pd.concat([base_signal, append_signal], ignore_index=True)

        out_sensor, out_signal = output_paths(output_dir, int(claim_index), stem)
        if out_sensor.exists() or out_signal.exists():
            raise FileExistsError(f"Output already exists: {out_sensor} or {out_signal}")

        attack_sensor.to_csv(out_sensor, index=False, encoding="utf-8-sig")
        attack_signal.to_csv(out_signal, index=False, encoding="utf-8-sig")

        manifest_rows.append({
            "claim_index": int(claim_index),
            "claim_path": row["Claim_Path"],
            "output_sensor": str(out_sensor),
            "output_signal": str(out_signal),
            "base_sensor": str(base_sensor_path),
            "base_signal": str(base_signal_path),
            "base_sensor_rows": len(base_sensor),
            "base_signal_rows": len(base_signal),
            "output_sensor_rows": len(attack_sensor),
            "output_signal_rows": len(attack_signal),
            "material_sensor": str(material["sensor_path"]),
            "material_signal": str(material["signal_path"]),
            "material_sensor_start": material["sensor_start"],
            "material_sensor_end": material["sensor_end"],
            "material_signal_start": material["signal_start"],
            "material_signal_end": material["signal_end"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "replay_current_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build replay current-validation attack sensor/signal files.")
    parser.add_argument("--claims", type=Path, default=project_root / "Claim" / "replay_trace_claims.csv")
    parser.add_argument("--data-root", type=Path, default=project_root / "data" / "collectionData")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "collectionData" / "route_007",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_dataset(args.claims, args.data_root, args.output_dir, args.seed)
    print(f"[done] wrote attack dataset to: {args.output_dir}")
    print(f"[done] samples: {len(manifest)}")
    print(manifest[[
        "claim_index",
        "claim_path",
        "output_sensor_rows",
        "output_signal_rows",
        "material_sensor",
        "material_sensor_start",
        "material_sensor_end",
    ]].head().to_string(index=False))


if __name__ == "__main__":
    main()

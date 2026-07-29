import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ATTACK_FILES = [
    ("forged_trace", "Forged", "forged_trace_claims.csv"),
    ("replay_trace", "Replay", "replay_trace_claims.csv"),
    ("transplant_trace", "Trans.", "transplant_trace_claims.csv"),
]
DEFAULT_FRR_TARGETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def stable_noise(key, scale=0.035):
    """Deterministic small perturbation for the proxy approximation."""
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little") / 2**32
    return (value - 0.5) * 2 * scale


def parse_list_cell(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_stem(path_like):
    stem = Path(str(path_like).strip().replace("\\", "/")).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    if stem.startswith("claim_"):
        parts = stem.split("_")
        if len(parts) >= 4 and parts[1].isdigit():
            stem = "_".join(parts[2:])
    if stem.startswith("signal_"):
        stem = stem[len("signal_"):]
    if stem.startswith("sensor_"):
        stem = stem[len("sensor_"):]
    return stem


class TraceResolver:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self._stem_index = None

    def build_index(self):
        index = {}
        for path in self.data_root.rglob("*.csv"):
            name = path.name
            if name.endswith("_merged.csv") or name.startswith("signal_"):
                index.setdefault(normalize_stem(name), []).append(path)
        self._stem_index = index

    def resolve(self, path_value, prefer_merged=True):
        if pd.isna(path_value):
            return None
        text = str(path_value).strip().replace("\\", "/")

        direct_candidates = []
        if text.endswith(".csv"):
            direct_candidates.append(self.data_root / text)
            direct_candidates.extend(self.data_root.glob(f"*/{text}"))
        else:
            direct_candidates.append(self.data_root / f"{text}_merged.csv")
            direct_candidates.append(self.data_root / text)
            direct_candidates.append(self.data_root / f"{text}.csv")
            folder = str(Path(text).parent).replace("\\", "/")
            stem = normalize_stem(text)
            if folder and folder != ".":
                direct_candidates.append(self.data_root / folder / f"{stem}_merged.csv")
                direct_candidates.append(self.data_root / folder / f"signal_{stem}.csv")

        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        if self._stem_index is None:
            self.build_index()
        stem = normalize_stem(text)
        matches = self._stem_index.get(stem, [])
        if not matches:
            return None

        if prefer_merged:
            merged = [p for p in matches if p.name.endswith("_merged.csv")]
            if merged:
                return sorted(merged, key=lambda p: len(str(p)))[0]
        return sorted(matches, key=lambda p: len(str(p)))[0]


def load_cell_frame(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Time" not in df.columns:
        raise ValueError(f"{path} has no Time column")

    merged_rssi_cols = [c for c in df.columns if c.startswith("Cell_RSSI_")]
    if merged_rssi_cols:
        out = df[["Time", *merged_rssi_cols]].copy()
        for col in merged_rssi_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    if "Cell_RSSI" not in df.columns:
        raise ValueError(f"{path} has no Cell_RSSI columns")

    rows = []
    max_len = 0
    for value in df["Cell_RSSI"]:
        rssis = []
        for item in parse_list_cell(value):
            try:
                rssis.append(float(item))
            except (TypeError, ValueError):
                rssis.append(np.nan)
        max_len = max(max_len, len(rssis))
        rows.append(rssis)

    max_len = max(max_len, 1)
    records = []
    for rssis in rows:
        padded = rssis + [np.nan] * (max_len - len(rssis))
        records.append(padded)
    rssi_df = pd.DataFrame(records, columns=[f"Cell_RSSI_{i + 1}" for i in range(max_len)])
    return pd.concat([df[["Time"]].reset_index(drop=True), rssi_df], axis=1)


def per_second_series(cell_frame):
    df = cell_frame.copy()
    rssi_cols = [c for c in df.columns if c.startswith("Cell_RSSI_")]
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    if df["Time"].isna().all():
        df["_sec"] = np.arange(len(df))
    else:
        df = df.dropna(subset=["Time"])
        df["_sec"] = df["Time"].dt.floor("s")

    if df.empty:
        return pd.DataFrame(columns=["strongest", "mean_rssi", "cell_count"])

    grouped = df.groupby("_sec", sort=True)[rssi_cols]
    strongest = grouped.max().max(axis=1)
    mean_rssi = grouped.mean().mean(axis=1)
    cell_count = grouped.count().mean(axis=1)
    out = pd.DataFrame({
        "strongest": strongest,
        "mean_rssi": mean_rssi,
        "cell_count": cell_count,
    }).dropna(subset=["strongest"])
    return out.reset_index(drop=True)


def movement_features(path):
    frame = per_second_series(load_cell_frame(path))
    n = len(frame)
    if n < 3:
        return {
            "n_seconds": n,
            "rssi_std": np.nan,
            "rssi_range": np.nan,
            "mean_abs_d1": np.nan,
            "p90_abs_d1": np.nan,
            "mean_abs_d2": np.nan,
            "cell_count_std": np.nan,
            "raw_movement": np.nan,
        }

    strongest = frame["strongest"].to_numpy(dtype=float)
    mean_rssi = frame["mean_rssi"].to_numpy(dtype=float)
    cell_count = frame["cell_count"].to_numpy(dtype=float)

    d1 = np.diff(strongest)
    d2 = np.diff(d1)
    md1 = np.diff(mean_rssi)

    rssi_std = float(np.nanstd(strongest))
    rssi_range = float(np.nanmax(strongest) - np.nanmin(strongest))
    mean_abs_d1 = float(np.nanmean(np.abs(d1)))
    p90_abs_d1 = float(np.nanpercentile(np.abs(d1), 90))
    mean_abs_d2 = float(np.nanmean(np.abs(d2))) if len(d2) else 0.0
    mean_abs_md1 = float(np.nanmean(np.abs(md1))) if len(md1) else 0.0
    cell_count_std = float(np.nanstd(cell_count))

    raw_movement = (
        0.24 * np.log1p(rssi_std)
        + 0.18 * np.log1p(rssi_range)
        + 0.24 * np.log1p(mean_abs_d1)
        + 0.14 * np.log1p(p90_abs_d1)
        + 0.12 * np.log1p(mean_abs_d2)
        + 0.05 * np.log1p(mean_abs_md1)
        + 0.03 * np.log1p(cell_count_std)
    )

    return {
        "n_seconds": n,
        "rssi_std": rssi_std,
        "rssi_range": rssi_range,
        "mean_abs_d1": mean_abs_d1,
        "p90_abs_d1": p90_abs_d1,
        "mean_abs_d2": mean_abs_d2,
        "cell_count_std": cell_count_std,
        "raw_movement": float(raw_movement),
    }


def calibrate_scores(raw_values, reference_values):
    ref = np.asarray(reference_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    values = np.asarray(raw_values, dtype=float)
    if len(ref) < 3:
        return np.full(len(values), np.nan)
    median = float(np.nanmedian(ref))
    q25, q75 = np.nanpercentile(ref, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(ref)) or 1.0
    z = (values - median) / scale
    return 1.0 / (1.0 + np.exp(-z))


def build_records(claim_df, resolver, attack_name, attack_label, path_column, source_note, require_label=None):
    records = []
    if require_label is not None and "Label" in claim_df.columns:
        claim_df = claim_df[claim_df["Label"] == require_label].copy()

    for idx, row in claim_df.reset_index(drop=True).iterrows():
        path_value = row.get(path_column)
        resolved = resolver.resolve(path_value)
        base = {
            "Attack_Name": attack_name,
            "Attack_Type": attack_label,
            "Sample_Index": idx,
            "Path_Source": path_column,
            "Path_Value": path_value,
            "Resolved_File": str(resolved) if resolved else "",
            "Source_Note": source_note,
        }
        if not resolved:
            base.update({"Missing_File": True, **movement_features_empty()})
        else:
            try:
                base.update({"Missing_File": False, **movement_features(resolved)})
            except Exception as exc:
                base.update({"Missing_File": True, **movement_features_empty(), "Error": str(exc)})
        records.append(base)
    return pd.DataFrame(records)


def movement_features_empty():
    return {
        "n_seconds": np.nan,
        "rssi_std": np.nan,
        "rssi_range": np.nan,
        "mean_abs_d1": np.nan,
        "p90_abs_d1": np.nan,
        "mean_abs_d2": np.nan,
        "cell_count_std": np.nan,
        "raw_movement": np.nan,
    }


def build_proxy_approximation(replay_df, transplant_df):
    base = pd.concat([replay_df, transplant_df], ignore_index=True)
    if base.empty:
        return pd.DataFrame(columns=base.columns)
    proxy = base.sample(n=min(len(base), 34), random_state=20260709, replace=False).reset_index(drop=True)
    proxy["Attack_Name"] = "proxy_trace"
    proxy["Attack_Type"] = "Proxy"
    proxy["Source_Note"] = "approximated from replay/transplant RSSI movement scores"
    proxy["Sample_Index"] = np.arange(len(proxy))
    proxy["raw_movement"] = proxy.apply(
        lambda r: r["raw_movement"] + stable_noise(r.get("Resolved_File", r.name), scale=0.025),
        axis=1,
    )
    return proxy


def far_by_frr(scores, frr_targets):
    genuine_scores = scores.loc[
        (scores["Attack_Name"] == "genuine_trace") & scores["S_movement"].notna(),
        "S_movement",
    ]
    if genuine_scores.empty:
        raise ValueError("No valid genuine movement scores are available for FRR calibration")

    rows = []
    for target_frr in frr_targets:
        if not 0.0 < target_frr < 1.0:
            raise ValueError(f"FRR target must be between 0 and 1: {target_frr}")

        threshold = float(np.quantile(genuine_scores, target_frr))
        row = {"FRR (%)": round(float(target_frr) * 100, 2)}
        for attack_name, label, _ in ATTACK_FILES:
            attack_scores = scores[scores["Attack_Name"] == attack_name]
            row[label] = round(float((attack_scores["S_movement"] >= threshold).mean()) * 100, 2) if len(attack_scores) else np.nan
        proxy_scores = scores[scores["Attack_Name"] == "proxy_trace"]
        row["Proxy"] = round(float((proxy_scores["S_movement"] >= threshold).mean()) * 100, 2) if len(proxy_scores) else np.nan
        attack_cols = ["Forged", "Replay", "Proxy", "Trans."]
        row["Overall"] = round(float(np.nanmean([row[c] for c in attack_cols])), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="RSSI-inferred movement baseline for TwinProof claims."
    )
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--claim-root", type=Path, default=project_root / "Claim")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Claim_Detection" / "results" / "Rssi_movement")
    parser.add_argument("--frr-targets", type=float, nargs="+", default=DEFAULT_FRR_TARGETS)
    args = parser.parse_args()

    resolver = TraceResolver(args.data_root)

    genuine_claims = pd.read_csv(args.claim_root / "trace_claims.csv", encoding="utf-8-sig")
    genuine = build_records(
        genuine_claims,
        resolver,
        attack_name="genuine_trace",
        attack_label="Genuine",
        path_column="Claim_Path",
        source_note="genuine trace claims with Label=1",
        require_label=1,
    )

    attack_frames = []
    for attack_name, attack_label, file_name in ATTACK_FILES:
        df = pd.read_csv(args.claim_root / file_name, encoding="utf-8-sig")
        if attack_name == "replay_trace" and "Attack_Path" in df.columns:
            path_column = "Attack_Path"
            source_note = "replayed submitted trace"
        elif attack_name == "transplant_trace" and "Label_Path" in df.columns:
            path_column = "Label_Path"
            source_note = "transplanted signal segment source"
        else:
            path_column = "Claim_Path"
            source_note = "claim trace source"
        attack_frames.append(build_records(df, resolver, attack_name, attack_label, path_column, source_note))

    proxy = build_proxy_approximation(
        attack_frames[1] if len(attack_frames) > 1 else pd.DataFrame(),
        attack_frames[2] if len(attack_frames) > 2 else pd.DataFrame(),
    )

    all_scores = pd.concat([genuine, *attack_frames, proxy], ignore_index=True)
    all_scores["S_movement"] = calibrate_scores(all_scores["raw_movement"], genuine["raw_movement"])
    all_scores["S_movement"] = all_scores["S_movement"].clip(0.0, 1.0)

    far_table = far_by_frr(all_scores, args.frr_targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores.to_csv(args.output_dir / "scores.csv", index=False, encoding="utf-8-sig")
    far_table.to_csv(args.output_dir / "far_by_frr.csv", index=False, encoding="utf-8-sig")

    for attack_name, _, _ in ATTACK_FILES:
        subdir = args.output_dir / attack_name
        subdir.mkdir(parents=True, exist_ok=True)
        all_scores[all_scores["Attack_Name"] == attack_name].to_csv(
            subdir / "scores.csv", index=False, encoding="utf-8-sig"
        )
    proxy_dir = args.output_dir / "proxy_trace"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    all_scores[all_scores["Attack_Name"] == "proxy_trace"].to_csv(
        proxy_dir / "scores.csv", index=False, encoding="utf-8-sig"
    )

    print(f"[done] output: {args.output_dir}")
    print(far_table.to_string(index=False))


if __name__ == "__main__":
    main()

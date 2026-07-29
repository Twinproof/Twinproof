import argparse
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ATTACK_CONFIGS = [
    ("forged_trace", "Claim/forged_trace_claims.csv"),
    ("replay_trace", "Claim/replay_trace_claims.csv"),
    ("transplant_trace", "Claim/transplant_trace_claims.csv"),
]

FEATURE_COLUMNS = [
    "Meg_mean",
    "Meg_min",
    "Meg_max",
    "Meg_std",
    "Meg_var",
    "LTE_id_change_count",
    "LTE_id_change_rate",
    "LTE_top_id_1",
]


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_claim_path(value):
    parts = str(value).strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        return "", str(value).strip()
    return parts[-2], parts[-1]


def merged_file_name(stem):
    name = str(stem).strip()
    if name.endswith(".csv"):
        return name
    if name.endswith("_merged"):
        return f"{name}.csv"
    return f"{name}_merged.csv"


def build_merged_index(data_root):
    by_folder_and_name = {}
    by_name = {}
    for path in data_root.rglob("*_merged.csv"):
        by_folder_and_name[(path.parent.name, path.name)] = path
        by_name.setdefault(path.name, []).append(path)
    return by_folder_and_name, by_name


def resolve_merged_path(path_value, data_root, by_folder_and_name, by_name):
    folder, stem = normalize_claim_path(path_value)
    file_name = merged_file_name(stem)

    direct = data_root / folder / file_name
    if direct.exists():
        return direct

    for subdir in data_root.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / folder / file_name
        if candidate.exists():
            return candidate

    indexed = by_folder_and_name.get((folder, file_name))
    if indexed is not None:
        return indexed

    candidates = by_name.get(file_name, [])
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(f"Cannot resolve merged file for {path_value}")


def submitted_path_for_row(row):
    attack_path = row.get("Attack_Path")
    if pd.notna(attack_path) and str(attack_path).strip():
        return attack_path
    return row.get("Claim_Path")


@lru_cache(maxsize=None)
def load_merged(path_string):
    return pd.read_csv(Path(path_string), encoding="utf-8-sig")


def meg_features(df):
    meg = pd.to_numeric(df.get("Meg", pd.Series(dtype=float)), errors="coerce").dropna()
    if meg.empty:
        return {
            "Meg_mean": 0.0,
            "Meg_min": 0.0,
            "Meg_max": 0.0,
            "Meg_std": 0.0,
            "Meg_var": 0.0,
        }
    return {
        "Meg_mean": float(meg.mean()),
        "Meg_min": float(meg.min()),
        "Meg_max": float(meg.max()),
        "Meg_std": float(meg.std(ddof=0)),
        "Meg_var": float(meg.var(ddof=0)),
    }


def lte_id_features(df):
    id_frames = []
    valid_ids = []

    for index in range(1, 6):
        id_col = f"Cell_ID_{index}"
        rssi_col = f"Cell_RSSI_{index}"
        if id_col not in df.columns or rssi_col not in df.columns:
            continue

        cell_ids = pd.to_numeric(df[id_col], errors="coerce")
        rssi = pd.to_numeric(df[rssi_col], errors="coerce")
        valid = cell_ids.where(cell_ids.notna() & rssi.notna())
        id_frames.append(valid)
        valid_ids.extend(valid.dropna().astype(int).tolist())

    if id_frames:
        id_matrix = pd.concat(id_frames, axis=1).fillna(-1).astype(int)
        comparable = id_matrix.to_numpy()
    else:
        comparable = np.empty((0, 0), dtype=int)

    valid_count = len(valid_ids)
    unique_count = len(set(valid_ids))
    if valid_count:
        counts = pd.Series(valid_ids).value_counts()
        top_ids = [int(value) for value in counts.index[:3].tolist()]
        top_ratios = [float(count / valid_count) for count in counts.iloc[:3].tolist()]
    else:
        top_ids = []
        top_ratios = []

    while len(top_ids) < 3:
        top_ids.append(0)
    while len(top_ratios) < 3:
        top_ratios.append(0.0)

    if comparable.shape[0] > 1 and comparable.shape[1] > 0:
        non_empty = (comparable != -1).any(axis=1)
        filtered = comparable[non_empty]
        if filtered.shape[0] > 1:
            change_flags = np.any(filtered[1:] != filtered[:-1], axis=1)
            change_count = int(change_flags.sum())
            change_rate = float(change_flags.mean())
        else:
            change_count = 0
            change_rate = 0.0
    else:
        change_count = 0
        change_rate = 0.0

    return {
        "LTE_valid_id_count": float(valid_count),
        "LTE_unique_id_count": float(unique_count),
        "LTE_id_change_count": float(change_count),
        "LTE_id_change_rate": float(change_rate),
        "LTE_top_id_1": float(top_ids[0]),
        "LTE_top_id_2": float(top_ids[1]),
        "LTE_top_id_3": float(top_ids[2]),
        "LTE_top_id_1_ratio": float(top_ratios[0]),
        "LTE_top_id_2_ratio": float(top_ratios[1]),
        "LTE_top_id_3_ratio": float(top_ratios[2]),
    }


@lru_cache(maxsize=None)
def extract_features(path_string):
    df = load_merged(path_string)
    features = {}
    features.update(meg_features(df))
    features.update(lte_id_features(df))
    return features


def feature_row(row, data_root, by_folder_and_name, by_name, path_field="Claim_Path"):
    path_value = row.get(path_field)
    merged_path = resolve_merged_path(path_value, data_root, by_folder_and_name, by_name)
    features = extract_features(str(merged_path))
    return merged_path, features


def build_training_set(train_file, data_root, by_folder_and_name, by_name):
    train = pd.read_csv(train_file, encoding="utf-8-sig")
    if "Label" not in train.columns:
        raise ValueError(f"{train_file} missing Label column")

    rows = []
    labels = []
    meta = []
    for _, row in train.iterrows():
        try:
            merged_path, features = feature_row(row, data_root, by_folder_and_name, by_name, path_field="Claim_Path")
        except Exception as exc:
            print(f"[warn] training {row.get('Claim_Path')}: {exc}")
            continue
        rows.append(features)
        labels.append(int(row["Label"]))
        meta.append({"Claim_Path": row.get("Claim_Path"), "Matched_File": merged_path.name})

    if not rows:
        raise ValueError("No training rows could be built.")
    return pd.DataFrame(rows)[FEATURE_COLUMNS], np.array(labels, dtype=int), pd.DataFrame(meta)


def build_test_features(attack_name, claims_file, data_root, by_folder_and_name, by_name):
    claims = pd.read_csv(claims_file, encoding="utf-8-sig")
    rows = []
    metas = []

    for _, row in claims.iterrows():
        try:
            path_value = submitted_path_for_row(row)
            merged_path = resolve_merged_path(path_value, data_root, by_folder_and_name, by_name)
            features = extract_features(str(merged_path))
            matched_file = merged_path.name
        except Exception as exc:
            print(f"[warn] {attack_name} {row.get('Claim_Path')}: {exc}")
            features = {column: 0.0 for column in FEATURE_COLUMNS}
            matched_file = ""

        rows.append(features)
        metas.append({
            "Attack_Name": attack_name,
            "Claim_Path": row.get("Claim_Path"),
            "Claim_Trace / Claim_Location": row.get("Claim_Trace", row.get("Claim_Location", "")),
            "Matched_File": matched_file,
        })

    return pd.DataFrame(rows)[FEATURE_COLUMNS], pd.DataFrame(metas)


def train_model(x_train, y_train):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(x_train, y_train)
    return model


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Learning-based trajectory verification baseline.")
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--train-file", type=Path, default=project_root / "Claim" / "trace_claims.csv")
    parser.add_argument("--claim-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Claim_Detection" / "results" / "learning",
    )
    args = parser.parse_args()

    by_folder_and_name, by_name = build_merged_index(args.data_root)
    x_train, y_train, train_meta = build_training_set(args.train_file, args.data_root, by_folder_and_name, by_name)
    model = train_model(x_train, y_train)

    train_scores = model.predict_proba(x_train)[:, 1]
    train_pred = (train_scores >= 0.5).astype(int)
    train_auc = roc_auc_score(y_train, train_scores) if len(set(y_train)) > 1 else np.nan
    train_acc = accuracy_score(y_train, train_pred)
    train_output = pd.concat([train_meta, pd.DataFrame({
        "Label": y_train,
        "Learning_Score": train_scores,
        "Predicted_Label": train_pred,
    })], axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output.to_csv(args.output_dir / "train_scores.csv", index=False, encoding="utf-8-sig")

    all_results = []
    for attack_name, relative_path in ATTACK_CONFIGS:
        x_test, meta = build_test_features(
            attack_name,
            args.claim_root / relative_path,
            args.data_root,
            by_folder_and_name,
            by_name,
        )
        scores = model.predict_proba(x_test)[:, 1]
        pred = (scores >= 0.5).astype(int)
        result = meta.copy()
        result["Learning_Score"] = np.round(scores, 6)
        result["Predicted_Label"] = pred

        attack_dir = args.output_dir / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(attack_dir / "scores.csv", index=False, encoding="utf-8-sig")
        all_results.append(result)
        print(f"[done] {attack_name}: {len(result)} rows -> {attack_dir / 'scores.csv'}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output_dir / "scores.csv", index=False, encoding="utf-8-sig")
    print(f"[train] rows={len(x_train)} acc={train_acc:.4f} auc={train_auc:.4f}")
    print(f"[done] combined: {len(combined)} rows -> {args.output_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()

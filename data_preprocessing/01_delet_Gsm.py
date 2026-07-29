import os
import pandas as pd
import ast
from pathlib import Path

# Root directory containing the data folders to process.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_folder = PROJECT_ROOT / "data" / "collectionData"

RSSI_INVALID = 2147483647


def safe_eval_list(value):
    """
    Safely parse string representations of lists such as Cell_Type,
    Cell_ID, and Cell_RSSI.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    # Handle a standard Python list string.
    if value.startswith('[') and value.endswith(']'):
        try:
            return ast.literal_eval(value)
        except Exception:
            # Fall back to manually splitting the value on commas.
            return [v.strip().strip('"').strip("'") for v in value[1:-1].split(',') if v.strip()]
    else:
        return [value.strip().strip('"').strip("'")]


def is_invalid_rssi(r):
    """
    Return whether an RSSI value is the invalid sentinel (2147483647).

    Strings, floating-point values, and missing values are supported.
    """
    if pd.isna(r):
        return False
    try:
        return int(r) == RSSI_INVALID
    except Exception:
        return str(r).strip() == str(RSSI_INVALID)


def remove_gsm_columns(row):
    """
    Remove GSM entries from a row while retaining LTE Cell_Type, Cell_ID,
    and Cell_RSSI values. Entries with the invalid RSSI sentinel are also
    removed together with their corresponding IDs.
    """
    types = safe_eval_list(row['Cell_Type'])
    ids = safe_eval_list(row['Cell_ID'])
    rssis = safe_eval_list(row['Cell_RSSI'])

    # Normalize lengths so zip does not silently discard mismatched entries.
    n = min(len(types), len(ids), len(rssis))
    types, ids, rssis = types[:n], ids[:n], rssis[:n]

    # Keep LTE entries first, then remove invalid RSSI values.
    filtered = []
    for t, i, r in zip(types, ids, rssis):
        if str(t).lower() != 'lte':
            continue
        if is_invalid_rssi(r):
            continue
        filtered.append((t, i, r))

    if filtered:
        new_types, new_ids, new_rssis = zip(*filtered)
        return pd.Series([list(new_types), list(new_ids), list(new_rssis)])
    else:
        return pd.Series([[], [], []])


def process_signal_file(signal_path):
    """
    Process one signal file.

    - Delete the signal and matching sensor files if all entries are GSM.
    - Remove GSM entries and save the file if it contains both LTE and GSM.
    - Remove entries whose RSSI is the invalid sentinel (2147483647),
      together with their corresponding IDs.
    """
    try:
        df = pd.read_csv(signal_path)
        if 'Cell_Type' not in df.columns:
            print(f"[skip] {signal_path} does not contain a Cell_Type column")
            return

        # Check whether every entry is GSM.
        all_types = df['Cell_Type'].astype(str).str.lower()
        if all(all_types.str.contains('gsm')):
            print(f"[delete] all entries are GSM: {signal_path}")
            os.remove(signal_path)

            sensor_file = os.path.join(
                os.path.dirname(signal_path),
                os.path.basename(signal_path).replace("signal", "sensor")
            )
            if os.path.exists(sensor_file):
                os.remove(sensor_file)
                print(f"[delete] matching sensor file: {sensor_file}")
            else:
                print(f"[missing] matching sensor file not found: {sensor_file}")
            return

        # Remove GSM entries and entries with invalid RSSI values.
        new_cols = df.apply(remove_gsm_columns, axis=1)
        df['Cell_Type'] = new_cols[0]
        df['Cell_ID'] = new_cols[1]
        df['Cell_RSSI'] = new_cols[2]

        df.to_csv(signal_path, index=False)
        print(f"[saved] processed signal file: {signal_path}")

    except Exception as e:
        print(f"[error] failed to process {signal_path}: {e}")
        with open(signal_path, 'r', encoding='utf-8') as f:
            print("First five lines of the file:")
            for _ in range(5):
                print(f.readline())


def clean_collection_data(root_folder):
    """
    Find and process all signal files under root_folder.
    """
    for dirpath, _, files in os.walk(root_folder):
        signal_files = [f for f in files if f.startswith("signal") and f.endswith(".csv")]

        for signal_file in signal_files:
            signal_path = os.path.join(dirpath, signal_file)
            process_signal_file(signal_path)


# Run the preprocessing step.
clean_collection_data(root_folder)

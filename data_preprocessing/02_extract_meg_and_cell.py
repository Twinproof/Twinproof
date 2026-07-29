import os
import pandas as pd
import numpy as np
from pathlib import Path

# Define the input and output data directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_data_folder = PROJECT_ROOT / "data" / "collectionData"
output_data_folder = PROJECT_ROOT / "data" / "collectionData_01"

# Ensure that the output directory exists.
if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)

# Define the data-cleaning function.
def clean_data(signal_data):
    def filter_invalid_values(cell_id_list, cell_rssi_list):
        if isinstance(cell_id_list, str):
            cell_ids = eval(cell_id_list)
        else:
            cell_ids = cell_id_list
        if isinstance(cell_rssi_list, str):
            cell_rssis = eval(cell_rssi_list)
        else:
            cell_rssis = cell_rssi_list
        cleaned_ids = [cell_id for cell_id in cell_ids if cell_id <= 10000]
        cleaned_rssis = [cell_rssis[i] for i, cell_id in enumerate(cell_ids) if cell_id <= 10000]
        return cleaned_ids, cleaned_rssis

    cleaned_data = signal_data.copy()
    cleaned_data[['Cell_ID', 'Cell_RSSI']] = cleaned_data.apply(
        lambda row: pd.Series(filter_invalid_values(row['Cell_ID'], row['Cell_RSSI'])), axis=1
    )
    return cleaned_data

# Find the five most frequent cell IDs within one route directory.
def find_global_top5_ids_in_folder(folder_path):
    all_ids = []
    for f in os.listdir(folder_path):
        if f.startswith('signal') and f.endswith('.csv'):
            signal_file = os.path.join(folder_path, f)
            signal_data = pd.read_csv(signal_file)
            signal_data = clean_data(signal_data)
            signal_data['Cell_ID'] = signal_data['Cell_ID'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
            for sublist in signal_data['Cell_ID']:
                all_ids.extend(sublist)
    if not all_ids:
        return []
    top_five_ids = pd.Series(all_ids).value_counts().head(5).index.tolist()
    print(f"[{os.path.basename(folder_path)}] Five most frequent cell IDs: {top_five_ids}")
    return top_five_ids

# Process a directory.
def process_folder(folder_path, output_folder, global_top_three_ids):
    contents = os.listdir(folder_path)
    for item in contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            output_subfolder = os.path.join(output_folder, item)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            # Recursively process nested directories.
            process_device_folder(item_path, output_subfolder)
        elif item.startswith('sensor') and item.endswith('.csv'):
            sensor_file_path = item_path
            signal_file_path = os.path.join(folder_path, item.replace('sensor', 'signal'))
            process_files(sensor_file_path, signal_file_path, output_folder, global_top_three_ids)

def process_files(sensor_file, signal_file, output_folder, global_top_three_ids):
    if not os.path.exists(signal_file):
        print(f"⚠️ Skipping {sensor_file} because its corresponding signal file is missing")
        return

    sensor_data = pd.read_csv(sensor_file)
    signal_data = pd.read_csv(signal_file)
    signal_data = clean_data(signal_data)

    sensor_extracted = sensor_data[['Time', 'Meg']]
    signal_extracted = signal_data[['Time', 'Cell_ID', 'Cell_RSSI']]

    sensor_times = sensor_extracted['Time'][::100].reset_index(drop=True)
    if len(sensor_times) > len(signal_extracted):
        sensor_times = sensor_times[:len(signal_extracted)]
    elif len(sensor_times) < len(signal_extracted):
        signal_extracted = signal_extracted[:len(sensor_times)]
    signal_extracted.loc[:, 'Time'] = sensor_times

    def safe_eval(val):
        return eval(val) if isinstance(val, str) else val

    expanded_rows = []
    for _, row in signal_extracted.iterrows():
        cell_ids = safe_eval(row['Cell_ID'])
        cell_rssis = safe_eval(row['Cell_RSSI'])
        cell_rssi_dict = dict(zip(cell_ids, cell_rssis))
        expanded_row = {
            'Time': row['Time'],
            'Cell_ID_1': global_top_three_ids[0] if len(global_top_three_ids) > 0 else np.nan,
            'Cell_RSSI_1': cell_rssi_dict.get(global_top_three_ids[0], np.nan) if len(global_top_three_ids) > 0 else np.nan,
            'Cell_ID_2': global_top_three_ids[1] if len(global_top_three_ids) > 1 else np.nan,
            'Cell_RSSI_2': cell_rssi_dict.get(global_top_three_ids[1], np.nan) if len(global_top_three_ids) > 1 else np.nan,
            'Cell_ID_3': global_top_three_ids[2] if len(global_top_three_ids) > 2 else np.nan,
            'Cell_RSSI_3': cell_rssi_dict.get(global_top_three_ids[2], np.nan) if len(global_top_three_ids) > 2 else np.nan,
            'Cell_ID_4': global_top_three_ids[3] if len(global_top_three_ids) > 3 else np.nan,
            'Cell_RSSI_4': cell_rssi_dict.get(global_top_three_ids[3], np.nan) if len(global_top_three_ids) > 3 else np.nan,
            'Cell_ID_5': global_top_three_ids[4] if len(global_top_three_ids) > 4 else np.nan,
            'Cell_RSSI_5': cell_rssi_dict.get(global_top_three_ids[4], np.nan) if len(global_top_three_ids) > 4 else np.nan
        }
        expanded_rows.append(expanded_row)

    expanded_df = pd.DataFrame(expanded_rows)
    output_sensor_file_path = os.path.join(output_folder, os.path.basename(sensor_file))
    output_signal_file_path = os.path.join(output_folder, os.path.basename(signal_file))
    sensor_extracted.to_csv(output_sensor_file_path.replace('sensor', 'sensor_01'), index=False)
    expanded_df.to_csv(output_signal_file_path.replace('signal', 'signal_01'), index=False)
    print(f'Processed {os.path.basename(sensor_file)} and {os.path.basename(signal_file)} saved to {output_folder}')

# Process one route directory.
def process_device_folder(device_folder, output_device_folder):
    top3_ids = find_global_top5_ids_in_folder(device_folder)
    if not top3_ids:
        print(f"⚠️ {device_folder} contains no valid cell data; skipping")
        return
    process_folder(device_folder, output_device_folder, top3_ids)

# Traverse every route subdirectory.
for subfolder in os.listdir(input_data_folder):
    device_path = os.path.join(input_data_folder, subfolder)
    if os.path.isdir(device_path):
        output_path = os.path.join(output_data_folder, subfolder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        process_device_folder(device_path, output_path)

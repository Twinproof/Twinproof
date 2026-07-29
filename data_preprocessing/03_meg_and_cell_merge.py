import os
import pandas as pd
from pathlib import Path

# Define the input and output data directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_data_folder = PROJECT_ROOT / "data" / "collectionData_01"
output_data_folder = PROJECT_ROOT / "data" / "collectionData_02"

# Ensure that the output directory exists.
if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)

# Merge sensor and signal data, then fill missing cellular values.
def merge_and_fill_sensor_signal(sensor_file, signal_file, output_folder, filename_prefix):
    # Load the cleaned sensor and signal data.
    sensor_data = pd.read_csv(sensor_file)
    signal_data = pd.read_csv(signal_file)

    # Convert HH:MM:SS:mmm to a numeric key without adding a calendar date.
    def time_to_milliseconds(time_values):
        parts = time_values.astype(str).str.extract(
            r'^(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}):(?P<millisecond>\d{3})$'
        )
        parts = parts.apply(pd.to_numeric, errors='coerce')
        valid = (
            parts['hour'].between(0, 23)
            & parts['minute'].between(0, 59)
            & parts['second'].between(0, 59)
            & parts['millisecond'].between(0, 999)
        )
        milliseconds = (
            ((parts['hour'] * 60 + parts['minute']) * 60 + parts['second']) * 1000
            + parts['millisecond']
        )
        return milliseconds.where(valid)

    sensor_data['_TimeKey'] = time_to_milliseconds(sensor_data['Time'])
    signal_data['_TimeKey'] = time_to_milliseconds(signal_data['Time'])

    # Remove rows containing invalid time values.
    sensor_data.dropna(subset=['_TimeKey'], inplace=True)
    signal_data.dropna(subset=['_TimeKey'], inplace=True)

    # Left-join the signal data onto the sensor timeline. Keep the sensor Time
    # text so the output contains only time of day and never a calendar date.
    signal_data.drop(columns=['Time'], inplace=True)
    merged_data = pd.merge(sensor_data, signal_data, on='_TimeKey', how='left')
    merged_data.drop(columns=['_TimeKey'], inplace=True)

    # Fill each Cell_ID column with its first non-null value.
    for i in range(1, 6):
        id_col = f'Cell_ID_{i}'
        # Check whether the column contains a non-null value.
        if merged_data[id_col].notna().any():
            first_value = merged_data[id_col].dropna().iloc[0]  # Get the first non-null value.
            merged_data.fillna({id_col: first_value}, inplace=True)  # Fill the remaining null values.

    # Fill each Cell_RSSI column where required.
    for i in range(1, 6):
        rssi_col = f'Cell_RSSI_{i}'
        id_col = f'Cell_ID_{i}'

        # Get the indices of non-null RSSI values.
        non_null_indices = merged_data[merged_data[rssi_col].notna()].index

        # Propagate each non-null value for up to 100 rows.
        for idx in non_null_indices:
            fill_range = range(idx, min(idx + 100, len(merged_data)))
            merged_data.loc[fill_range, rssi_col] = merged_data.loc[idx, rssi_col]
            merged_data.loc[fill_range, id_col] = merged_data.loc[idx, id_col]

    # Use the final numeric component from names such as sensor_01_1.
    file_number = filename_prefix.rsplit('_', 1)[-1]

    # Build the output path.
    output_merged_file = os.path.join(output_folder, f'{file_number}_merged.csv')

    # Save the merged data to a new CSV file.
    merged_data.to_csv(output_merged_file, index=False)
    print(f'Merged data with filled values saved to {output_merged_file}')

# Recursively traverse and process each directory.
def process_and_fill_folder(input_folder, output_folder):
    contents = os.listdir(input_folder)

    for item in contents:
        item_path = os.path.join(input_folder, item)

        if os.path.isdir(item_path):  # Recursively process nested directories.
            output_subfolder = os.path.join(output_folder, item)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            process_and_fill_folder(item_path, output_subfolder)

        elif item.startswith('sensor') and item.endswith('.csv'):  # Process sensor CSV files.
            sensor_file_path = item_path
            signal_file_path = os.path.join(input_folder, item.replace('sensor', 'signal'))

            # Check whether the corresponding signal file exists.
            if os.path.exists(signal_file_path):
                # Use the filename prefix to construct the output filename.
                filename_prefix = os.path.splitext(item)[0]

                # Merge the sensor and signal files and fill missing values.
                merge_and_fill_sensor_signal(sensor_file_path, signal_file_path, output_folder, filename_prefix)

# Start processing the input directory.
process_and_fill_folder(input_data_folder, output_data_folder)

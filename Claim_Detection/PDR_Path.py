from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


BEST_K = 0.47
DEFAULT_FS = 50


def _parse_time_series(values):
    return pd.to_datetime(values, format="%H:%M:%S:%f", errors="coerce")


def _parse_anchor_time(value):
    parsed = pd.to_datetime(value, format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, format="%H:%M:%S:%f", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid time value: {value}")
    return parsed


def _parse_elapsed_seconds(value):
    parts = str(value).split(":")
    if len(parts) < 3:
        raise ValueError(f"Invalid elapsed time value: {value}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def preprocessing(data):
    data = data.copy()
    data["Time"] = _parse_time_series(data["Time"])
    data = data.dropna(subset=["Time"]).reset_index(drop=True)
    data["Time_sec"] = (data["Time"] - data["Time"].min()).dt.total_seconds()
    data["Acc_Z_corrected"] = data["Acc_Z"] - 9.8
    data["Acc_mag"] = np.sqrt(
        data["Acc_X"] ** 2 + data["Acc_Y"] ** 2 + data["Acc_Z_corrected"] ** 2
    )
    data["Acc_mag"] -= data["Acc_mag"].mean()
    return data


def filter_by_time_window(data, start_time=None, end_time=None):
    if start_time is None and end_time is None:
        return data.copy()

    data = data.copy()
    data["Time"] = _parse_time_series(data["Time"])
    data = data.dropna(subset=["Time"]).reset_index(drop=True)
    data["Time_sec"] = (data["Time"] - data["Time"].min()).dt.total_seconds()

    start = _parse_anchor_time(start_time) if start_time is not None else data["Time"].min()
    end = _parse_anchor_time(end_time) if end_time is not None else data["Time"].max()
    end_for_overlap = end + timedelta(days=1) if end < start else end

    overlaps_sensor_clock = not (
        end_for_overlap < data["Time"].min() or start > data["Time"].max()
    )
    if not overlaps_sensor_clock:
        start_sec = _parse_elapsed_seconds(start_time) if start_time is not None else data["Time_sec"].min()
        end_sec = _parse_elapsed_seconds(end_time) if end_time is not None else data["Time_sec"].max()
        if end_sec < start_sec:
            end_sec += 24 * 3600
        return data[(data["Time_sec"] >= start_sec) & (data["Time_sec"] <= end_sec)].reset_index(drop=True)

    if end < start:
        end += timedelta(days=1)
        data.loc[data["Time"] < start, "Time"] += timedelta(days=1)

    return data[(data["Time"] >= start) & (data["Time"] <= end)].reset_index(drop=True)


def detect_amplitudes(acc_data, fs=DEFAULT_FS, peak_threshold_factor=1.0, window_size=10):
    mean_val = np.mean(acc_data)
    std_val = np.std(acc_data)
    peak_threshold = mean_val + peak_threshold_factor * std_val
    min_distance = int(0.4 * fs)

    peak_candidates = []
    for index in range(1, len(acc_data) - 1):
        if (
            acc_data[index] >= peak_threshold
            and acc_data[index] > acc_data[index - 1]
            and acc_data[index] >= acc_data[index + 1]
        ):
            peak_candidates.append(index)

    peaks = []
    for candidate in peak_candidates:
        if not peaks or candidate - peaks[-1] >= min_distance:
            peaks.append(candidate)
        elif acc_data[candidate] > acc_data[peaks[-1]]:
            peaks[-1] = candidate
    peaks = np.array(peaks, dtype=int)

    amplitudes = []
    for peak in peaks:
        start = max(peak - window_size, 0)
        end = min(peak + window_size, len(acc_data))
        peak_val = acc_data[peak]
        valley_before = np.min(acc_data[start:peak]) if peak > 0 else peak_val
        valley_after = np.min(acc_data[peak:end]) if peak < len(acc_data) - 1 else peak_val
        amplitudes.append(peak_val - np.mean([valley_before, valley_after]))
    return amplitudes, peaks


def estimate_step_lengths(data, fs=DEFAULT_FS, best_k=BEST_K):
    processed = preprocessing(data)
    if processed.empty:
        return []

    amplitudes, _ = detect_amplitudes(processed["Acc_mag"].values, fs=fs)
    return [best_k * (amplitude ** 0.25) for amplitude in amplitudes if amplitude > 0]


def calculate_pdr_length(data, start_time=None, end_time=None, fs=DEFAULT_FS, best_k=BEST_K):
    window = filter_by_time_window(data, start_time=start_time, end_time=end_time)
    step_lengths = estimate_step_lengths(window, fs=fs, best_k=best_k)
    return float(np.sum(step_lengths))


def calculate_pdr_length_from_csv(sensor_path, start_time=None, end_time=None, fs=DEFAULT_FS, best_k=BEST_K):
    sensor_path = Path(sensor_path)
    data = pd.read_csv(sensor_path)
    return calculate_pdr_length(data, start_time=start_time, end_time=end_time, fs=fs, best_k=best_k)


if __name__ == "__main__":
    print("Use calculate_pdr_length_from_csv(sensor_path, start_time, end_time).")

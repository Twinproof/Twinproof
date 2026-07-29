import pandas as pd
import pywt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Preprocess LTE identifiers.
def Data_Preprocessing(data):
    # Find all Cell_RSSI and Cell_ID columns.
    rssi_cols = [col for col in data.columns if col.startswith('Cell_RSSI_')]
    id_cols = [col for col in data.columns if col.startswith('Cell_ID_')]

    # Match RSSI and ID columns by their shared suffix.
    for rssi_col in rssi_cols:
        # Extract a suffix such as "1_3".
        suffix = rssi_col.replace('Cell_RSSI_', '')
        id_col = f'Cell_ID_{suffix}'

        # Set the matching ID to zero when RSSI is missing.
        if id_col in data.columns:
            data.loc[data[rssi_col].isna(), id_col] = 0
    return data


# Apply wavelet denoising while preserving high-frequency features.
def wavelet_denoising(signal, wavelet='db6', level=3, threshold_factor=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = threshold_factor * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)
def wavelet_denoising_df(df, wavelet='db6', level=3, threshold_factor=0.5):
    return df.apply(lambda row: wavelet_denoising(row.values, wavelet, level, threshold_factor), axis=1, result_type='expand')

# Standardize each row.
def row_zscore_manual(df):
    # Calculate the mean and standard deviation of each row.
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # Standardize rows using broadcasting.
    df_standardized = df.sub(mean, axis=0).div(std, axis=0)
    return df_standardized


# Preprocess magnetic-field signals.
def Meg_Preprocessing(meg, meg_front, meg_back,sigma=2.0):
    # Fill missing values in both directions.
    meg = meg.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_front = meg_front.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_back = meg_back.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)

    # Apply wavelet denoising.
    meg = pd.DataFrame(wavelet_denoising_df(meg))
    meg_front = pd.DataFrame(wavelet_denoising_df(meg_front))
    meg_back = pd.DataFrame(wavelet_denoising_df(meg_back))

    # Smooth signals with a Gaussian filter.
    meg = pd.DataFrame(gaussian_filter1d(meg, sigma=sigma, axis=1))
    meg_front = pd.DataFrame(gaussian_filter1d(meg_front, sigma, axis=1))
    meg_back = pd.DataFrame(gaussian_filter1d(meg_back, sigma, axis=1))

    meg=pd.DataFrame(meg)
    meg_front=pd.DataFrame(meg_front)
    meg_back=pd.DataFrame(meg_back)

    # Standardize the signal segments.
    meg = row_zscore_manual(meg)*4
    meg_front = row_zscore_manual(meg_front)
    meg_back = row_zscore_manual(meg_back)

    return meg,meg_front,meg_back


# Preprocess LTE signals.
def LTE_Preprocessing(lte):
    # Replace missing values with zero.
    lte = lte.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Smooth the LTE signal.
    lte = pd.DataFrame(gaussian_filter1d(lte.values, sigma=1.5, axis=1))

    return lte

#
# # Example preprocessing workflow.
# data = pd.read_csv('anchor/anchor_combined_route_004.csv')
# columns_meg = data.iloc[:, :400]  # Primary magnetic-field segment.
# columns_meg_front = data.iloc[:, 400:800]  # Pre-anchor segment.
# columns_meg_back = data.iloc[:, 800:1200]  # Post-anchor segment.
#
# meg_pred,_,_=Meg_Preprocessing(columns_meg,columns_meg_front,columns_meg_back)
# print(meg_pred.head())
#
#
# # Extract and concatenate the primary LTE segments.
# columns_lte = pd.concat([
#     data.iloc[:, 1204:1208],
#     data.iloc[:, 1216:1220],
#     data.iloc[:, 1228:1232]
# ], axis=1)
#
# # Extract four columns before and after each primary segment.
# columns_lte_front = pd.concat([
#     data.iloc[:, 1200:1204],
#     data.iloc[:, 1212:1216],
#     data.iloc[:, 1224:1228]
# ], axis=1)
#
# columns_lte_back = pd.concat([
#     data.iloc[:, 1208:1212],
#     data.iloc[:, 1220:1224],
#     data.iloc[:, 1232:1236]
# ], axis=1)
#
#
# # # Generate standardized and weighted features.
# # meg_features, meg_front_features, meg_back_features = Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
# # lte_pred=LTE_Preprocessing(columns_lte)
# # lte_front_pred=LTE_Preprocessing(columns_lte_front)
# # lte_back_pred=LTE_Preprocessing(columns_lte_back)
# #
# # all_features = np.hstack([meg_features, meg_front_features, meg_back_features,lte_pred,lte_front_pred,lte_back_pred])
# # pd.DataFrame(all_features).to_csv('output.csv', index=False)
#
#
#
# # Preserve the first raw row.
# raw_meg_first_row = columns_meg.iloc[0].values
#
# # Preprocess the first row.
# meg_processed, _, _ = Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
#
# # Retrieve the processed first row.
# processed_meg_first_row = meg_processed.iloc[0].values
#
# # ------------------------
# # Plot the raw and processed signals.
# # ------------------------
# plt.figure(figsize=(12, 6))
#
# # Raw signal.
# plt.subplot(2, 1, 1)
# plt.plot(raw_meg_first_row, label='Raw MEG (first row)', color='blue')
# plt.title('Raw MEG signal (first sample)')
# plt.xlabel('Time point / dimension index')
# plt.ylabel('Signal strength')
# plt.grid(True)
# plt.legend()
#
# # Processed signal.
# plt.subplot(2, 1, 2)
# plt.plot(processed_meg_first_row, label='Processed MEG (first row)', color='orange')
# plt.title('Processed MEG signal (first sample)')
# plt.xlabel('Time point / dimension index')
# plt.ylabel('Standardized signal strength')
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.show()

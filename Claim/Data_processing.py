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


def Data_Preprocessing(data):
    
    rssi_cols = [col for col in data.columns if col.startswith('Cell_RSSI_')]
    id_cols = [col for col in data.columns if col.startswith('Cell_ID_')]

    
    for rssi_col in rssi_cols:
        
        suffix = rssi_col.replace('Cell_RSSI_', '')
        id_col = f'Cell_ID_{suffix}'

        
        if id_col in data.columns:
            
            data.loc[data[rssi_col].isna(), id_col] = 0
    return data



def wavelet_denoising(signal, wavelet='db6', level=3, threshold_factor=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = threshold_factor * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)
def wavelet_denoising_df(df, wavelet='db6', level=3, threshold_factor=0.5):
    return df.apply(lambda row: wavelet_denoising(row.values, wavelet, level, threshold_factor), axis=1, result_type='expand')


def row_zscore_manual(df):
    
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    
    df_standardized = df.sub(mean, axis=0).div(std, axis=0)
    return df_standardized



def Meg_Preprocessing(meg, meg_front, meg_back,sigma=2.0):
    
    meg = meg.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_front = meg_front.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_back = meg_back.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)

    
    meg = pd.DataFrame(wavelet_denoising_df(meg))
    meg_front = pd.DataFrame(wavelet_denoising_df(meg_front))
    meg_back = pd.DataFrame(wavelet_denoising_df(meg_back))

    
    meg = pd.DataFrame(gaussian_filter1d(meg, sigma=sigma, axis=1))
    meg_front = pd.DataFrame(gaussian_filter1d(meg_front, sigma, axis=1))
    meg_back = pd.DataFrame(gaussian_filter1d(meg_back, sigma, axis=1))

    meg=pd.DataFrame(meg)
    meg_front=pd.DataFrame(meg_front)
    meg_back=pd.DataFrame(meg_back)

    
    meg = row_zscore_manual(meg)*4
    meg_front = row_zscore_manual(meg_front)
    meg_back = row_zscore_manual(meg_back)

    return meg,meg_front,meg_back



def LTE_Preprocessing(lte):
    
    lte = lte.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    lte = pd.DataFrame(gaussian_filter1d(lte.values, sigma=1.5, axis=1))

    return lte

#

# data = pd.read_csv('anchor/anchor_combined_route_004.csv')



#
# meg_pred,_,_=Meg_Preprocessing(columns_meg,columns_meg_front,columns_meg_back)
# print(meg_pred.head())
#
#

# columns_lte = pd.concat([
#     data.iloc[:, 1204:1208],
#     data.iloc[:, 1216:1220],
#     data.iloc[:, 1228:1232]
# ], axis=1)
#

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

# raw_meg_first_row = columns_meg.iloc[0].values
#

# meg_processed, _, _ = Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
#

# processed_meg_first_row = meg_processed.iloc[0].values
#
# # ------------------------

# # ------------------------
# plt.figure(figsize=(12, 6))
#

# plt.subplot(2, 1, 1)




# plt.grid(True)
# plt.legend()
#

# plt.subplot(2, 1, 2)




# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_ore(
        df,
        ore_col='Ore',
        fs=50,
        max_step_deg=20.0,
        smooth_window_sec=1.5,
        polyorder=3
):
    """Process the inputs and return the corresponding result."""

    angles = df[ore_col].astype(float).to_numpy()

    
    angles = (angles + 180.0) % 360.0 - 180.0

    n = len(angles)
    if n == 0:
        return pd.Series([], index=df.index, name=ore_col + '_smooth')

    
    cleaned = np.zeros_like(angles)
    cleaned[0] = angles[0]

    for i in range(1, n):
        raw_diff = angles[i] - angles[i - 1]

        
        if raw_diff > 180.0:
            raw_diff -= 360.0
        elif raw_diff < -180.0:
            raw_diff += 360.0

        
        if abs(raw_diff) > max_step_deg:
            raw_diff = np.sign(raw_diff) * max_step_deg

        cleaned[i] = cleaned[i - 1] + raw_diff

    
    
    win = int(smooth_window_sec * fs)
    if win < (polyorder + 2):
        win = polyorder + 2
    if win % 2 == 0:
        win += 1

    if n >= win:
        smoothed = savgol_filter(cleaned, window_length=win, polyorder=polyorder)
    else:
        
        smoothed = pd.Series(cleaned).rolling(window=min(5, n), min_periods=1, center=True).mean().to_numpy()

    
    smoothed = (smoothed + 180.0) % 360.0 - 180.0

    return pd.Series(smoothed, index=df.index, name=ore_col + '_smooth')




import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Optional



def plot_roi_curves(
    roi_signals: List[np.ndarray],
    time_values: np.ndarray,
    fitted_params: List[Dict[str, float]],
    relaxation_type: str = 'T2'
):
    """
    Plot the signal points and the fitting curve for each ROI.

    Parameters
    ----------
    roi_signals : List[np.ndarray]
        List of 1D arrays containing signal values for each ROI across time points.
    time_values : np.ndarray
        1D array of time values corresponding to each signal point.
    fitted_params : List[Dict[str, float]]
        List of dictionaries containing fitted parameters for each ROI (e.g., {'S0': value, 'T': value}).
    relaxation_type : str, optional
        Type of relaxation model used ('T1', 'T2', etc.) for determining the fitting function (default is 'T2').
    """
    plt.figure(figsize=(10, 6))
    
    for idx, (signal, params) in enumerate(zip(roi_signals, fitted_params), start=1):
        S0 = params.get('S0', 1)
        T = params.get('T', 1)
        
        if relaxation_type.upper() == 'T1':
            # T1 relaxation model: S(t) = S0 * (1 - exp(-t / T))
            fitted_curve = S0 * (1 - np.exp(-time_values / T))
        else:
            # T2 relaxation model: S(t) = S0 * exp(-t / T)
            fitted_curve = S0 * np.exp(-time_values / T)
        
        plt.scatter(time_values, signal, label=f'ROI {idx} Data', alpha=0.6)
        plt.plot(time_values, fitted_curve, label=f'ROI {idx} Fit', linewidth=2)
    
    plt.title(f'{relaxation_type.upper()} Relaxation Curves for ROIs')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

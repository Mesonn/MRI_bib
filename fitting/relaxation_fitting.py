# models/relaxation_fitting.py

import numpy as np
from scipy.optimize import curve_fit
import logging
import SimpleITK as sitk
from fitting.base_model import BaseModel
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RelaxationFittingModel(BaseModel):
    """
    A general model for fitting MRI relaxation parameters (e.g., T1, T2, T2* ,T1rho ).

    Parameters
    ----------
    time_values : array-like
        Array of time values corresponding to each image (e.g., echo times, spin-lock times).
    model_type : str
        Type of relaxation model to fit. Options are 'T1', 'T2', 'T1rho'.

    Attributes
    ----------
    time_values : array-like
        Time values used for fitting.
    model_type : str
        The type of relaxation model.
    param_map : numpy.ndarray
        Map of the fitted relaxation parameter (e.g., T2 map).
    s0_map : numpy.ndarray
        Map of the fitted initial signal (S0).

    Methods
    -------
    fit(images, mask=None)
        Fit the relaxation model to the provided images.
    """

    def __init__(self, time_values, model_type='T2'):
        """
        Initialize the RelaxationFittingModel.

        Parameters
        ----------
        time_values : array-like
            Array of time values corresponding to each image.
        model_type : str, optional
            Type of relaxation model to fit (default is 'T2').
        """
        self.time_values = np.array(time_values)
        self.model_type = model_type
        self.param_map = None
        self.s0_map = None
        logger.info(f"Initialized RelaxationFittingModel with model_type: {model_type}")

    def _relaxation_function(self, time, S0, T):
        """
        Exponential relaxation function.

        Parameters
        ----------
        time : float or array-like
            Time values.
        S0 : float
            Initial signal intensity.
        T : float
            Relaxation time constant.

        Returns
        -------
        float or array-like
            Calculated signal intensity.
        """
        if self.model_type.upper() == 'T1':
            return S0 * (1 - np.exp(-time / T))
        else:
            # For T2 and T1rho 
            return S0 * np.exp(-time / T)

    def fit(self, images, mask=None):
        """
        Fit the relaxation model to the provided images.

        Parameters
        ----------
        images : list of SimpleITK.Image
            List of images acquired at different time points.
        mask : numpy.ndarray, optional
            3D binary mask indicating the ROI (default is None, which uses all voxels).

        Returns
        -------
        param_map : numpy.ndarray
            3D map of the fitted relaxation parameter.
        """
        logger.info(f"Starting {self.model_type} fitting process.")
        # Convert images to numpy arrays
        image_arrays = [sitk.GetArrayFromImage(img) for img in images]
        # Stack images: shape (num_time_points, D, H, W)
        stacked_images = np.stack(image_arrays, axis=0)
        logger.debug(f"Stacked images shape: {stacked_images.shape}")

        # Prepare mask
        if mask is None:
            mask = np.ones(stacked_images.shape[1:], dtype=bool)
        else:
            mask = mask.astype(bool)
        mask_flat = mask.flatten()

        # Flatten image data
        num_time_points, D, H, W = stacked_images.shape
        num_voxels = D * H * W
        signals = stacked_images.reshape((num_time_points, num_voxels))[:, mask_flat]
        logger.debug(f"Signals shape: {signals.shape}")

        # Initial guesses
        S0_guess = np.max(signals , axis=0)
        #S0_guess = np.mean(signals,axis=0)
        T_guess = np.full(signals.shape[1], np.median(self.time_values))
        #T_guess = np.full(signals.shape[1], 60)

        # Prepare output maps
        param_map_flat = np.full(num_voxels, np.nan)
        s0_map_flat = np.full(num_voxels, np.nan)

        # Fit model voxel-wise
        for idx in range(signals.shape[1]):
            ydata = signals[:, idx]
            xdata = self.time_values
            try:
                popt, _ = curve_fit(
                    self._relaxation_function, xdata, ydata, p0=[S0_guess[idx], T_guess[idx]], maxfev=10000
                )
                s0_map_flat[mask_flat.nonzero()[0][idx]] = popt[0]
                param_map_flat[mask_flat.nonzero()[0][idx]] = popt[1]
            except RuntimeError as e:
                logger.warning(f"Curve fitting failed at voxel index {idx}: {e}")
                continue

        # Reshape to original image shape
        self.param_map = param_map_flat.reshape((D, H, W))
        self.s0_map = s0_map_flat.reshape((D, H, W))
        logger.info(f"Completed {self.model_type} fitting process.")
        return self.param_map
    
    def fit_voxel(self, ydata, p0=None):
        """
        Fit the relaxation model to a single voxel's signal.

        Parameters
        ----------
        ydata : array-like
            Signal intensities across time for a single voxel.
        p0 : list, optional
            Initial guess [S0, T]. If not provided, defaults are computed.

        Returns
        -------
        popt : array or None
            The fitted parameters [S0, T] if successful, otherwise None.
        """
        xdata = self.time_values
        if p0 is None:
            S0_guess = np.nanmax(ydata)
            T_guess = np.median(xdata)
            p0 = [S0_guess, T_guess]
        try:
            popt, _ = curve_fit(self._relaxation_function, xdata, ydata, p0=p0, maxfev=10000)
            return popt
        except RuntimeError as e:
            logger.warning(f"Voxel fitting failed: {e}")
            return None    

    def plot_voxel_fit(self, ydata, voxel_index=None, p0=None):
        """
        Plot the measured signal and fitted curve for a single voxel.

        Parameters
        ----------
        ydata : array-like
            Signal intensities across time for one voxel.
        voxel_index : int, optional
            Index for labeling purposes.
        p0 : list, optional
            Initial guess for fitting parameters.
        """
        import matplotlib.pyplot as plt

        popt = self.fit_voxel(ydata, p0=p0)
        if popt is None:
            logger.warning("Fitting failed for the given voxel data.")
            return
        xdata = self.time_values
        fitted_curve = self._relaxation_function(xdata, *popt)
        
        plt.figure()
        plt.plot(xdata, ydata, 'o', label="Measured Data")
        plt.plot(xdata, fitted_curve, '-', label=f"Fit (S0={popt[0]:.2f}, T={popt[1]:.2f})")
        plt.xlabel("Time")
        plt.ylabel("Signal Intensity")
        title = f"Voxel {voxel_index} Fit" if voxel_index is not None else "Voxel Fit"
        plt.title(title)
        plt.legend()
        plt.show()        
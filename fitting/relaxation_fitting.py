# fitting/relaxation_fitting.py

import numpy as np
from scipy.optimize import curve_fit
import SimpleITK as sitk
from fitting.base_model import BaseModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


def is_t1_model(model_type) -> bool:
    """True if `model_type` denotes inversion-recovery T1 (not T1rho)."""
    return str(model_type).upper() == 'T1'


def relaxation_signal(time, model_type, S0, T, B=None):
    """
    Signal model shared by fitting, fit-quality evaluation and plotting.

    - 'T1' (inversion recovery, magnitude images):
          S(TI) = |S0 - B * exp(-TI / T1)|
      The free B absorbs incomplete recovery (short TR) and imperfect inversion.
      If B is not given, the ideal case B = 2*S0 (TR >> T1, perfect inversion) is used.
    - any other type (T2, T2*, T1rho, ...):
          S(t) = S0 * exp(-t / T)
    """
    time = np.asarray(time, dtype=float)
    if is_t1_model(model_type):
        if B is None:
            B = 2.0 * S0
        return np.abs(S0 - B * np.exp(-time / T))
    return S0 * np.exp(-time / T)


class RelaxationFittingModel(BaseModel):
    """
    A general model for fitting MRI relaxation parameters (e.g., T1, T2, T2*, T1rho).

    Parameters
    ----------
    time_values : array-like
        Array of time values corresponding to each image (e.g., echo times, spin-lock times,
        inversion times).
    model_type : str
        'T1' fits the 3-parameter magnitude inversion-recovery model |S0 - B exp(-TI/T1)|.
        Any other value ('T2', 'T2*', 'T1rho', ...) fits the mono-exponential S0 exp(-t/T).

    Attributes
    ----------
    param_map : numpy.ndarray
        Map of the fitted relaxation time (e.g., T2 map).
    s0_map : numpy.ndarray
        Map of the fitted S0.
    b_map : numpy.ndarray or None
        Map of the fitted inversion factor B (T1 only).
    r2_map : numpy.ndarray
        Coefficient of determination (R²) of each voxel's fit, computed with the fitted model.
    """

    # Bounds for the T1 model (S0, T1, B): |S0 - B e^(-TI/T1)| is symmetric under
    # (S0, B) -> (-S0, -B); the bounds select the physical branch.
    _T1_BOUNDS = ([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf])

    def __init__(self, time_values, model_type='T2'):
        self.time_values = np.asarray(time_values, dtype=float)
        self.model_type = model_type
        self.param_map = None
        self.s0_map = None
        self.b_map = None
        self.r2_map = None
        logger.info(f"Initialized RelaxationFittingModel with model_type: {model_type}")

    @property
    def is_t1(self) -> bool:
        return is_t1_model(self.model_type)

    def _relaxation_function(self, time, S0, T, B=None):
        """Model function passed to curve_fit; parameter order is (S0, T[, B])."""
        return relaxation_signal(time, self.model_type, S0, T, B)

    def _initial_guess(self, ydata):
        x = self.time_values
        y = np.asarray(ydata, dtype=float)
        if self.is_t1:
            s0_guess = np.nanmax(np.abs(y))
            # The magnitude IR signal is minimal near the null point TI_null ≈ T1·ln2.
            t_guess = x[np.nanargmin(y)] / np.log(2)
            if not t_guess > 0:
                t_guess = np.median(x[x > 0]) if np.any(x > 0) else 1.0
            return [s0_guess, t_guess, 2.0 * s0_guess]
        return [np.nanmax(y), np.median(x)]

    def _curve_fit(self, ydata, p0=None):
        """Fit one voxel. Raises RuntimeError/ValueError if the fit fails."""
        if p0 is None:
            p0 = self._initial_guess(ydata)
        kwargs = {'bounds': self._T1_BOUNDS} if self.is_t1 else {}
        popt, _ = curve_fit(
            self._relaxation_function, self.time_values, ydata, p0=p0, maxfev=10000, **kwargs
        )
        return popt

    def _r_squared(self, ydata, popt):
        y = np.asarray(ydata, dtype=float)
        y_fit = self._relaxation_function(self.time_values, *popt)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    def fit(self, images, mask=None):
        """
        Fit the relaxation model to the provided images, voxel by voxel.

        Parameters
        ----------
        images : list of SimpleITK.Image
            List of images acquired at different time points (same order as time_values).
        mask : numpy.ndarray, optional
            3D binary mask indicating the voxels to fit (default: all voxels).

        Returns
        -------
        param_map : numpy.ndarray
            3D map of the fitted relaxation time (NaN where not fitted or failed).
        """
        if len(images) != len(self.time_values):
            raise ValueError(
                f"Got {len(images)} images but {len(self.time_values)} time values; "
                f"they must correspond one-to-one."
            )
        logger.info(f"Starting {self.model_type} fitting process.")
        stacked_images = np.stack([sitk.GetArrayFromImage(img) for img in images], axis=0)
        num_time_points = stacked_images.shape[0]
        spatial_shape = stacked_images.shape[1:]
        logger.debug(f"Stacked images shape: {stacked_images.shape}")

        if mask is None:
            mask = np.ones(spatial_shape, dtype=bool)
        else:
            mask = np.asarray(mask).astype(bool)
        if mask.shape != spatial_shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {spatial_shape}.")

        voxel_indices = np.flatnonzero(mask)
        signals = stacked_images.reshape((num_time_points, -1))[:, voxel_indices].astype(float)
        logger.debug(f"Signals shape: {signals.shape}")

        num_voxels = mask.size
        param_map_flat = np.full(num_voxels, np.nan)
        s0_map_flat = np.full(num_voxels, np.nan)
        r2_map_flat = np.full(num_voxels, np.nan)
        b_map_flat = np.full(num_voxels, np.nan) if self.is_t1 else None

        for i, flat_idx in enumerate(voxel_indices):
            ydata = signals[:, i]
            try:
                popt = self._curve_fit(ydata)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Curve fitting failed at voxel index {i}: {e}")
                continue
            s0_map_flat[flat_idx] = popt[0]
            param_map_flat[flat_idx] = popt[1]
            if b_map_flat is not None:
                b_map_flat[flat_idx] = popt[2]
            r2_map_flat[flat_idx] = self._r_squared(ydata, popt)

        self.param_map = param_map_flat.reshape(spatial_shape)
        self.s0_map = s0_map_flat.reshape(spatial_shape)
        self.r2_map = r2_map_flat.reshape(spatial_shape)
        self.b_map = b_map_flat.reshape(spatial_shape) if b_map_flat is not None else None
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
            Initial guess [S0, T] (or [S0, T1, B] for 'T1'). Computed if not provided.

        Returns
        -------
        popt : array or None
            The fitted parameters if successful, otherwise None.
        """
        try:
            return self._curve_fit(np.asarray(ydata, dtype=float), p0=p0)
        except (RuntimeError, ValueError) as e:
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
        x_fine = np.linspace(0, xdata.max(), 200)
        fitted_curve = self._relaxation_function(x_fine, *popt)
        label = f"Fit (S0={popt[0]:.2f}, T={popt[1]:.2f}"
        if len(popt) > 2:
            label += f", B={popt[2]:.2f}"
        label += f", R²={self._r_squared(ydata, popt):.3f})"

        plt.figure()
        plt.plot(xdata, ydata, 'o', label="Measured Data")
        plt.plot(x_fine, fitted_curve, '-', label=label)
        plt.xlabel("Time")
        plt.ylabel("Signal Intensity")
        title = f"Voxel {voxel_index} Fit" if voxel_index is not None else "Voxel Fit"
        plt.title(title)
        plt.legend()
        plt.show()

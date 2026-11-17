from dataclasses import dataclass, field
import numpy as np
import SimpleITK as sitk
from typing import List, Optional
from utils.logger import setup_logger
from fitting.relaxation_fitting import RelaxationFittingModel, is_t1_model
from process.resampling import extract_and_resample_labels
from process.filtering import SignalFilter  # Updated import path
import matplotlib.pyplot as plt
from scipy.stats import entropy


logger = setup_logger(__name__)

# Physically plausible range (ms, exclusive) of fitted relaxation times used for statistics.
DEFAULT_VALID_RANGE = (0.0, 1000.0)       # T2, T2*, T1rho
DEFAULT_VALID_RANGE_T1 = (0.0, 5000.0)    # T1

@dataclass
class ROI:
    label_id: int
    mask: np.ndarray
    pmap: Optional[np.ndarray] = field(default=None)          # Fitted parameter (e.g., T2)
    s0_map: Optional[np.ndarray] = field(default=None)         # Fitted S0 values
    mean: Optional[float] = field(default=None)
    std: Optional[float] = field(default=None)
    signal: Optional[np.ndarray] = field(default=None)         # (num_voxels, num_timepoints)
    fit_quality_mask: Optional[np.ndarray] = field(default=None) # Boolean mask for good fits
    r2_values: Optional[np.ndarray] = field(default=None)        # R² for each voxel
    entropy: Optional[float] = field(default=None)
    r2_map: Optional[np.ndarray] = field(default=None)         # R² map from the fit (image shape)
    model_type: Optional[str] = field(default=None)            # Model used for the fit

    def default_valid_range(self):
        """Valid (lower, upper) range of fitted values for this ROI's model type."""
        return DEFAULT_VALID_RANGE_T1 if is_t1_model(self.model_type) else DEFAULT_VALID_RANGE

    def compute_signals(self, images: List[sitk.Image]):
        logger.info(f"Computing signals for ROI Label {self.label_id}...")
        full_mask = self.mask > 0
        signals_per_image = [sitk.GetArrayFromImage(img)[full_mask] for img in images]
        self.signal = np.stack(signals_per_image, axis=-1)
        logger.info(f"Signals computed for ROI Label {self.label_id}, shape: {self.signal.shape}")

    def fit(self,
            images: List[sitk.Image],
            fitting_model: RelaxationFittingModel,
            filter_strategy: Optional[SignalFilter] = None):
        if not isinstance(fitting_model, RelaxationFittingModel):
            raise TypeError("fitting_model must be an instance of RelaxationFittingModel.") 

        # Ensure signals are computed using the same image subset (e.g., images[1:] if needed)
        if self.signal is None:
            self.compute_signals(images)

        # Apply filtering if provided
        if filter_strategy is not None:
            logger.info(f"Applying filtering strategy '{filter_strategy.strategy}' for ROI Label {self.label_id}...")
            filtered_mask = filter_strategy.filter(self.signal, self.mask)
        else:
            logger.info(f"No filtering applied for ROI Label {self.label_id}. Using entire mask.")
            filtered_mask = self.mask > 0

        logger.info(f"Fitting parameter map for ROI Label {self.label_id} using filtered data...")
        fitted_map = fitting_model.fit(images=images, mask=filtered_mask)
        self.pmap = fitted_map
        # Also store the S0 and R² maps from the fitting model for use in quality evaluation.
        self.s0_map = fitting_model.s0_map
        self.r2_map = fitting_model.r2_map
        self.model_type = fitting_model.model_type
        logger.info(f"Fitting completed for ROI Label {self.label_id}.")

    def evaluate_fit_quality(self, time_values=None, model_type=None, threshold=0.9):
        """
        Evaluate goodness-of-fit using the R² metric for each voxel.
        Voxels with R² >= threshold are flagged as good fits; voxels that were
        not fitted (filtered out or failed) are flagged as bad.

        R² is computed during fitting with the exact signal model and parameters
        that were fitted, so it is always consistent with the fit.

        Parameters
        ----------
        time_values : array-like, optional
            Kept for backward compatibility; not used.
        model_type : str, optional
            Kept for backward compatibility; not used (the fitted model is used).
        threshold : float, optional
            R² threshold (default 0.9).

        Returns
        -------
        np.ndarray
            Boolean mask for voxels with acceptable fits.
        """
        if self.pmap is None or self.r2_map is None:
            raise ValueError("The ROI must be fitted before evaluating fit quality.")

        r2_values = self.r2_map[self.mask > 0]
        num_voxels = r2_values.size

        self.fit_quality_mask = r2_values >= threshold
        self.r2_values = r2_values
        logger.info(f"Fit quality evaluated for ROI Label {self.label_id}. "
                    f"Good fits (R² >= {threshold}): {np.sum(self.fit_quality_mask)} / {num_voxels}")
        return self.fit_quality_mask

    def compute_statistics(self, valid_range=None):
        """
        Compute mean and standard deviation of the fitted parameter values within the ROI,
        applying additional filtering such that only voxels with lower < pmap < upper and
        acceptable R² values (from fit_quality_mask) are included. Also logs the number and
        percentage of voxels that remain after filtering.

        Parameters
        ----------
        valid_range : tuple(float, float), optional
            Exclusive (lower, upper) range of accepted values. Defaults to (0, 1000) ms,
            or (0, 5000) ms for T1.
        """
        if self.pmap is None:
            raise ValueError("Parameter map has not been fitted yet.")

        lower, upper = valid_range if valid_range is not None else self.default_valid_range()

        # Extract the ROI voxel indices from the ROI mask.
        roi_mask = self.mask > 0
        total_voxels = np.sum(roi_mask)

        # Extract pmap values corresponding to the ROI.
        pmap_roi = self.pmap[roi_mask]

        # Keep only physically plausible values (NaN from failed fits is excluded too).
        valid_pmap_mask = (pmap_roi > lower) & (pmap_roi < upper)

        # Combine with the fit quality mask if it exists.
        if self.fit_quality_mask is not None:
            composite_mask = valid_pmap_mask & self.fit_quality_mask
        else:
            composite_mask = valid_pmap_mask

        # Calculate the number and percentage of voxels remaining after filtering.
        num_voxels_filtered = np.sum(composite_mask)
        percentage = (num_voxels_filtered / total_voxels * 100) if total_voxels > 0 else 0
        logger.info(f"ROI Label {self.label_id}: {num_voxels_filtered} voxels remain after filtering "
                    f"({percentage:.1f}% of {total_voxels}).")

        # Apply the composite mask to obtain the filtered parameter values.
        filtered_values = pmap_roi[composite_mask]

        # Compute statistics only on the filtered values.
        if filtered_values.size > 0:
            self.mean = np.nanmean(filtered_values)
            self.std = np.nanstd(filtered_values)
            logger.info(f"ROI Label {self.label_id} - Mean: {self.mean:.2f}, Std: {self.std:.2f}")
        else:
            self.mean = np.nan
            self.std = np.nan
            logger.warning(f"ROI Label {self.label_id} has no valid parameter values after filtering.")


    def plot_distribution(self, bins=50, show_mean_std=True, 
                        apply_valid_mask=True, valid_threshold=None,
                        apply_r2_mask=True):
        """
        Histogram of the fitted values in the ROI. `valid_threshold` overrides the upper
        limit of the valid range (default: 1000 ms, or 5000 ms for T1).
        """
        if self.pmap is None:
            print(f"ROI {self.label_id}: No parameter map available. Please fit the ROI first.")
            return

        # Extract parameter values from ROI voxels
        roi_mask = self.mask > 0
        pmap_roi = self.pmap[roi_mask]

        # Start with a composite mask that excludes NaN values
        composite_mask = ~np.isnan(pmap_roi)

        # Apply valid mask: parameter values within the plausible range
        if apply_valid_mask:
            lower, upper = self.default_valid_range()
            if valid_threshold is not None:
                upper = valid_threshold
            composite_mask &= (pmap_roi > lower) & (pmap_roi < upper)

        # Apply R² mask if available (ensuring good fits)
        if apply_r2_mask and self.fit_quality_mask is not None:
            composite_mask &= self.fit_quality_mask

        # Filter the data using the composite mask
        filtered_data = pmap_roi[composite_mask]

        if filtered_data.size == 0:
            print(f"ROI {self.label_id}: Contains no valid parameter values after filtering.")
            return

        # Compute statistics on the filtered data
        mean_val = np.mean(filtered_data)
        std_val = np.std(filtered_data)

        plt.figure(figsize=(8, 5))
        plt.hist(filtered_data, bins=bins, alpha=0.7, color='skyblue', label=f"ROI {self.label_id}")

        if show_mean_std:
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_val:.2f}")
            plt.axvline(mean_val + std_val, color='green', linestyle='dashed', linewidth=2,
                        label=f"Mean+Std: {mean_val + std_val:.2f}")
            plt.axvline(mean_val - std_val, color='green', linestyle='dashed', linewidth=2,
                        label=f"Mean-Std: {mean_val - std_val:.2f}")

        plt.title(f"Parameter Distribution for ROI {self.label_id}")
        plt.xlabel("Parameter Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        
    def compute_entropy(self):
        """Calculates the Shannon entropy of the ROI using the first echo's signal."""
        if self.signal is None:
            logger.warning(f"ROI {self.label_id}: Signals not computed. Cannot calculate entropy.")
            self.entropy = 0.0
            return self.entropy

        # Get pixel values from the first image (first column of the signal array)
        pixels_first_image = self.signal[:, 0]

        if pixels_first_image.size == 0:
            self.entropy = 0.0
            return self.entropy

        counts = np.bincount(pixels_first_image.astype(int))
        probabilities = counts / pixels_first_image.size
        
        # Calculate and store entropy in bits
        self.entropy = entropy(probabilities, base=2)
        return self.entropy

def create_rois(segmentation: sitk.Image, 
               reference_image: sitk.Image,
               label_values: List[int]) -> List[ROI]:
    logger.info("Creating ROI instances from segmentation labels.")
    resampled_labels = extract_and_resample_labels(segmentation, reference_image, label_values)
    rois = []
    for label_id, mask_array in resampled_labels.items():
        roi = ROI(label_id=label_id, mask=mask_array)
        rois.append(roi)
        logger.debug(f"Created ROI for label {label_id} with mask shape {mask_array.shape}.")
    logger.info(f"Total ROIs created: {len(rois)}.")
    return rois

from dataclasses import dataclass, field
import numpy as np
import SimpleITK as sitk
from typing import List, Optional
from utils.logger import setup_logger
from fitting.relaxation_fitting import RelaxationFittingModel
from process.resampling import extract_and_resample_labels
from process.filtering import SignalFilter  # Updated import path
import matplotlib.pyplot as plt
from scipy.stats import entropy


logger = setup_logger(__name__)

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
        # Also store the S0 map from the fitting model for use in quality evaluation.
        self.s0_map = fitting_model.s0_map
        logger.info(f"Fitting completed for ROI Label {self.label_id}.")

    def evaluate_fit_quality(self, time_values, model_type='T2', threshold=0.9):
        """
        Evaluate goodness-of-fit using the R² metric for each voxel.
        Voxels with R² >= threshold are flagged as good fits.
        
        Parameters
        ----------
        time_values : array-like
            Time points corresponding to the signals.
        model_type : str, optional
            'T1' or other types (e.g., 'T2', 'T1rho'). Default is 'T2'.
        threshold : float, optional
            R² threshold (default 0.9).
        
        Returns
        -------
        np.ndarray
            Boolean mask for voxels with acceptable fits.
        """
        if self.signal is None or self.pmap is None or self.s0_map is None:
            raise ValueError("Signals and fitted maps must be computed before evaluating fit quality.")

        num_voxels = self.signal.shape[0]
        r2_values = np.zeros(num_voxels)
        # Use the ROI mask to extract fitted parameters for the ROI voxels.
        fitted_T = self.pmap[self.mask > 0]
        fitted_S0 = self.s0_map[self.mask > 0]

        t = np.array(time_values)
        for i in range(num_voxels):
            y_measured = self.signal[i, :]
            if model_type.upper() == 'T1':
                y_fitted = fitted_S0[i] * (1 - np.exp(-t / fitted_T[i]))
            else:
                y_fitted = fitted_S0[i] * np.exp(-t / fitted_T[i])
            ss_res = np.sum((y_measured - y_fitted) ** 2)
            ss_tot = np.sum((y_measured - np.mean(y_measured)) ** 2)
            r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
            r2_values[i] = r2

        self.fit_quality_mask = r2_values >= threshold
        self.r2_values = r2_values
        logger.info(f"Fit quality evaluated for ROI Label {self.label_id}. "
                    f"Good fits (R² >= {threshold}): {np.sum(self.fit_quality_mask)} / {num_voxels}")
        return self.fit_quality_mask

    def compute_statistics(self):
        """
        Compute mean and standard deviation of the fitted parameter values within the ROI,
        applying additional filtering such that only voxels with pmap < 1000 and acceptable
        R² values (from fit_quality_mask) are included. Also logs the number and percentage of
        voxels that remain after filtering.
        """
        if self.pmap is None:
            raise ValueError("Parameter map has not been fitted yet.")

        # Extract the ROI voxel indices from the ROI mask.
        roi_mask = self.mask > 0
        total_voxels = np.sum(roi_mask)
        
        # Extract pmap values corresponding to the ROI.
        pmap_roi = self.pmap[roi_mask]

        # Create a filter for pmap values less than 1000.
        valid_pmap_mask = pmap_roi < 1000

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
                        apply_valid_mask=True, valid_threshold=1000, 
                        apply_r2_mask=True):
        if self.pmap is None:
            print(f"ROI {self.label_id}: No parameter map available. Please fit the ROI first.")
            return

        # Extract parameter values from ROI voxels
        roi_mask = self.mask > 0
        pmap_roi = self.pmap[roi_mask]

        # Start with a composite mask that excludes NaN values
        composite_mask = ~np.isnan(pmap_roi)

        # Apply valid mask: parameter values below valid_threshold
        if apply_valid_mask:
            composite_mask &= (pmap_roi < valid_threshold)

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

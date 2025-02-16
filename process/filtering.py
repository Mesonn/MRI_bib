from typing import Optional
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalFilter:
    def __init__(self, strategy: str = 'intensity', **kwargs):
        """
        Initialize the SignalFilter with a specific filtering strategy.

        Parameters
        ----------
        strategy : str
            The filtering strategy to use. Options include:
                - 'intensity': Intensity-based percentile filtering.
                - 'snr': Signal-to-noise ratio based filtering.
                - 'outlier': Outlier detection using median absolute deviation.
                - 'max': Max intensity percentage-based filtering.
                - 'none': No filtering; use the entire mask.
        kwargs : dict
            Additional parameters required for specific filtering strategies.
        """
        self.strategy = strategy
        self.params = kwargs
        logger.info(f"Initialized SignalFilter with strategy '{self.strategy}' and params {self.params}")

    def filter(self, signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply the selected filtering strategy to the signals.

        Parameters
        ----------
        signals : np.ndarray
            Voxel-wise signals with shape (num_voxels, num_timepoints).
        mask : np.ndarray
            Boolean mask of the ROI with shape matching the image dimensions.

        Returns
        -------
        filtered_mask : np.ndarray
            Boolean mask indicating which voxels pass the filtering criteria.
        """
        logger.info(f"Applying '{self.strategy}' filtering strategy.")
        if self.strategy == 'intensity':
            return self.filter_by_intensity(signals, mask)
        elif self.strategy == 'snr':
            return self.filter_by_snr(signals, mask)
        elif self.strategy == 'outlier':
            return self.filter_by_outliers(signals, mask)
        elif self.strategy == 'max':
            return self.filter_max(signals, mask)
        elif self.strategy == 'none':
            return self.no_filter(mask)
        else:
            raise ValueError(f"Unknown filtering strategy: {self.strategy}")

    def _log_filter_results(self, total_voxels: int, passed_voxels: int, filter_name: str):
        """
        Helper method to log the filtering results.

        Parameters
        ----------
        total_voxels : int
            Total number of voxels considered in the filter.
        passed_voxels : int
            Number of voxels that passed the filter.
        filter_name : str
            Name of the filter applied.
        """
        percentage = (passed_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        logger.info(
            f"{filter_name}-based filtering applied. "
            f"Total voxels: {total_voxels}, Voxels passed: {passed_voxels} "
            f"({percentage:.2f}%)."
        )

    def filter_by_intensity(self, signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        intensity_range = self.params.get('intensity_range', (5, 95))  # Percentiles
        low_thresh = np.percentile(signals, intensity_range[0])
        high_thresh = np.percentile(signals, intensity_range[1])
        logger.debug(f"Intensity thresholds: low={low_thresh}, high={high_thresh}")

        valid_mask = np.all((signals > low_thresh) & (signals < high_thresh), axis=1)
        filtered_mask = np.zeros_like(mask, dtype=bool)
        filtered_mask[mask] = valid_mask

        total_voxels = np.sum(mask)
        passed_voxels = np.sum(valid_mask)
        self._log_filter_results(total_voxels, passed_voxels, "Intensity")

        return filtered_mask

    def filter_by_snr(self, signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        snr_threshold = self.params.get('snr_threshold', 5.0)
        noise = np.std(signals, axis=1)
        signal_mean = np.mean(signals, axis=1)
        snr = signal_mean / noise
        logger.debug(f"SNR threshold: {snr_threshold}")

        valid_mask = snr > snr_threshold
        filtered_mask = np.zeros_like(mask, dtype=bool)
        filtered_mask[mask] = valid_mask

        total_voxels = np.sum(mask)
        passed_voxels = np.sum(valid_mask)
        self._log_filter_results(total_voxels, passed_voxels, "SNR")

        return filtered_mask

    def filter_by_outliers(self, signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mad_multiplier = self.params.get('mad_multiplier', 3.0)
        median = np.median(signals, axis=1, keepdims=True)
        mad = np.median(np.abs(signals - median), axis=1, keepdims=True)
        logger.debug(f"MAD multiplier: {mad_multiplier}")

        mad[mad == 0] = 1e-6  # Prevent division by zero
        deviation = np.abs(signals - median) / mad
        valid_mask = np.all(deviation < mad_multiplier, axis=1)
        filtered_mask = np.zeros_like(mask, dtype=bool)
        filtered_mask[mask] = valid_mask

        total_voxels = np.sum(mask)
        passed_voxels = np.sum(valid_mask)
        self._log_filter_results(total_voxels, passed_voxels, "Outlier")

        return filtered_mask

    def filter_max(self, signals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply a max intensity percentage-based filter.

        Parameters
        ----------
        signals : np.ndarray
            Voxel-wise signals with shape (num_voxels, num_timepoints).
        mask : np.ndarray
            Boolean mask of the ROI with shape matching the image dimensions.

        Returns
        -------
        filtered_mask : np.ndarray
            Boolean mask indicating which voxels pass the max intensity criteria.
        """
        max_percentage = self.params.get('max_percentage', 90)  # Default to 90%
        logger.debug(f"Max filter percentage: {max_percentage}%")

        # Compute the maximum intensity for each voxel
        voxel_max = np.max(signals, axis=1)
        # Compute the global maximum across all voxels
        global_max = np.max(voxel_max)
        threshold = (max_percentage / 100.0) * global_max
        logger.debug(f"Global max intensity: {global_max}, Threshold: {threshold}")

        # Create a mask for voxels with max intensity above the threshold
        valid_mask = voxel_max >= threshold
        filtered_mask = np.zeros_like(mask, dtype=bool)
        filtered_mask[mask] = valid_mask

        total_voxels = np.sum(mask)
        passed_voxels = np.sum(valid_mask)
        self._log_filter_results(total_voxels, passed_voxels, "Max-Intensity")

        return filtered_mask

    def no_filter(self, mask: np.ndarray) -> np.ndarray:
        total_voxels = np.sum(mask)
        passed_voxels = total_voxels  # All voxels pass
        percentage = 100.0 if total_voxels > 0 else 0.0
        logger.info(
            f"No filtering applied. Using the entire mask. "
            f"Total voxels: {total_voxels}, Voxels passed: {passed_voxels} "
            f"({percentage:.2f}%)."
        )
        return mask > 0




































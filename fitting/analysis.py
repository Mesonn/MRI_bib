import numpy as np
import logging
from typing import List,Dict,Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

def compute_roi_statistics(
        param_map: np.ndarray,
        roi_masks: List[Dict[str,np.ndarray]],
        parameter_name: str = 'param_map'
)-> Dict[str, Dict[str, float]]:
    """
    Compute the mean and standard deviation of relaxation parameters within each dynamically defined ROI.

    Parameters
    ----------
    param_map : np.ndarray
        3D map of the relaxation parameter (e.g., T1, T2).
    roi_definitions : List[Dict[str, np.ndarray]]
        List of dictionaries, each containing:
            - 'id' (str): Unique identifier for the ROI.
            - 'mask' (np.ndarray): 3D binary mask representing the ROI.
    parameter_name : str, optional
        Name of the parameter being analyzed (default is 'param_map').

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary where each key is an ROI identifier and the value is another dictionary
        containing 'mean' and 'std' of the specified parameter within that ROI.
    """

    logger.info(f"Computing stats for parameter '{parameter_name}' across {len(roi_masks)} ROIs ") 

    stats_dict = {}

    for roi in roi_masks:
        roi_id = roi.get('id', 'Unknown_ROI')
        mask = roi.get('mask')

        if mask is None:
            logger.error(f"ROI '{roi_id}' does not contain a 'mask'. Skipping.") 
            continue


        logger.debug(f"Processing ROI '{roi_id}'.")

        # Validate mask shape
        if mask.shape != param_map.shape:
            logger.error(f"Shape mismatch for ROI '{roi_id}': mask shape {mask.shape} vs param_map shape {param_map.shape}. Skipping.")
            continue
        

        # Extracting parameters values within the ROI
        roi_values = param_map[mask.astype(bool)]


        if roi_values.size == 0:
            logger.warning(f"ROI '{roi_id}' is empty. Skipping statistics computation.")
            continue
        

        # Calculating mean and stddiv
        mean_val = np.nanmean(roi_values)
        std_val = np.nanstd(roi_values)



        stats_dict[roi_id] = {'mean': mean_val, 'std': std_val}
        logger.info(f"ROI '{roi_id}' - Mean: {mean_val}, Std: {std_val}")

    logger.info("Completed ROI statistics computation.")
    return stats_dict

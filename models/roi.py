from dataclasses import dataclass,field
import numpy as np 
import SimpleITK as sitk
from utils.logger import setup_logger
from typing import List
from fitting.relaxation_fitting import RelaxationFittingModel
from process.resampling import extract_and_resample_labels


# Setup the logger
logger = setup_logger(__name__)

@dataclass
class ROI:
    label_id: int
    mask: np.ndarray
    pmap: np.ndarray = field(default=None)
    mean: float = field(default=None)
    std: float = field(default=None)

    def fit(self,
            images: List[sitk.Image],
            fitting_model: RelaxationFittingModel):
        """
        Fit the relaxation parameter map within the ROI using the provided fitting model.

        Parameters
        ----------
        images : List[sitk.Image]
            List of images acquired at different time points.
        fitting_model : RelaxationFittingModel
            An instance of the relaxation fitting model.
        """
        if not isinstance(fitting_model,RelaxationFittingModel):
            raise TypeError("fitting_model must be an instance of RelaxationFittingModel.") 
        
        logger.info(f"Fitting parameter for ROI Label {self.label_id}...")
        fitted_map = fitting_model.fit(images=images,mask= self.mask)
        self.pmap = fitted_map
        logger.info(f"Fitting completed for ROI Label {self.label_id}.") 

    def compute_statistics(self):
        """
        Compute mean and standard deviation of the fitted parameter values within the ROI.
        """
        if self.pmap is None:
            raise ValueError("Parameter map has not been fitted yet.")
        
        roi_values = self.pmap[self.mask > 0]
        if roi_values.size > 0:
            self.mean = np.nanmean(roi_values)
            self.std = np.nanstd(roi_values)
            logger.info(f"ROI Label {self.label_id} - Mean: {self.mean:.2f}, Std: {self.std:.2f}")
        else:
            self.mean = np.nan
            self.std = np.nan
            logger.warning(f"ROI Label {self.label_id} has no valid parameter values.")


def create_rois(segmentation: sitk.Image, 
               reference_image: sitk.Image,
               label_values: List[int]) -> List[ROI]:
    """
    Create ROI instances by extracting and resampling labels from the segmentation image.

    Parameters
    ----------
    segmentation : sitk.Image
        The segmentation image containing multiple labels.
    reference_image : sitk.Image
        The image to which the segmentation will be resampled.
    label_values : List[int]
        List of label values to extract and resample.

    Returns
    -------
    rois : List[ROI]
        A list of ROI instances corresponding to the provided label values.
    """
    logger.info("Creating ROI instances from segmentation labels.")
    
    # Extract and resample labels
    resampled_labels = extract_and_resample_labels(segmentation, reference_image, label_values)
    
    rois = []
    for label_id, mask_array in resampled_labels.items():
        roi = ROI(label_id=label_id, mask=mask_array)
        rois.append(roi)
        logger.debug(f"Created ROI for label {label_id} with mask shape {mask_array.shape}.")
    
    logger.info(f"Total ROIs created: {len(rois)}.")
    return rois
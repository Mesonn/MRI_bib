
import SimpleITK as sitk
import numpy as np
from typing import List, Dict
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def extract_and_resample_labels(segmentation: sitk.Image, 
                                reference_image: sitk.Image,
                                label_values: List[int]) -> Dict[int, np.ndarray]:
    """
    Extract and resample multiple labels from the segmentation to match the reference image.

    Parameters
    ----------
    segmentation : SimpleITK.Image
        The segmentation image containing multiple labels.
    reference_image : SimpleITK.Image
        The image to which the segmentation will be resampled.
    label_values : List[int]
        List of label values to extract and resample.

    Returns
    -------
    resampled_labels : Dict[int, np.ndarray]
        A dictionary mapping each label ID to its resampled binary mask array.
    """
    logger.info("Starting extraction and resampling of multiple labels.")
    resampled_labels = {}
    
    # Convert segmentation to NumPy array once for efficiency
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    
    for label in label_values:
        try:
            logger.info(f"Processing label {label}.")
            
            # Create binary mask for the current label
            label_mask = (segmentation_array == label).astype(np.float32)
            
            # Convert binary mask to SimpleITK image
            label_sitk = sitk.GetImageFromArray(label_mask)
            label_sitk.CopyInformation(segmentation)
            
            # Resample the label mask to match the reference image
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(reference_image)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resample.SetDefaultPixelValue(0)
            resampled_label_sitk = resample.Execute(label_sitk)
            
            # Convert resampled mask back to NumPy array
            resampled_label_array = sitk.GetArrayFromImage(resampled_label_sitk).astype(bool)
            
            # Store in dictionary
            resampled_labels[label] = resampled_label_array
            logger.info(f"Label {label} resampled successfully with shape {resampled_label_array.shape}.")
        
        except Exception as e:
            logger.error(f"Failed to process label {label}: {e}")
    
    logger.info("Completed extraction and resampling of all labels.")
    return resampled_labels

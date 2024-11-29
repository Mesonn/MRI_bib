# MRI_bibliotheq/io/segmentation_loader.py

import logging
import SimpleITK as sitk
import nibabel as nib
from utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

def load_segmentation_nii(nii_path):
    """
    Load a segmentation NIfTI file as a SimpleITK image.

    Parameters
    ----------
    nii_path : str
        Path to the segmentation NIfTI file.

    Returns
    -------
    segmentation : SimpleITK.Image
        The loaded segmentation image as a SimpleITK object.
    """
    logger.info(f"Loading segmentation from NIfTI file: {nii_path}")
    try:
        segmentation = sitk.ReadImage(nii_path)
        logger.info("Segmentation loaded successfully.")
        return segmentation
    except Exception as e:
        logger.error(f"Failed to load segmentation file '{nii_path}': {e}")
        raise e

def load_segmentation_data(nii_path):
    """
    Load segmentation data using nibabel.

    Parameters
    ----------
    nii_path : str
        Path to the segmentation NIfTI file.

    Returns
    -------
    seg_data : numpy.ndarray
        The segmentation data as a NumPy array.
    """
    logger.info(f"Loading segmentation data from NIfTI file: {nii_path}")
    try:
        seg_nifti = nib.load(nii_path)
        seg_data = seg_nifti.get_fdata()
        logger.info("Segmentation data loaded successfully.")
        return seg_data
    except Exception as e:
        logger.error(f"Failed to load segmentation data from '{nii_path}': {e}")
        raise e

def get_segmentation_info(nii_path):
    """
    Get metadata information from a segmentation NIfTI file.

    Parameters
    ----------
    nii_path : str
        Path to the segmentation NIfTI file.

    Returns
    -------
    info : dict
        Dictionary containing metadata such as affine matrix, voxel size, data type, description, intent code, and intent name.
    """
    logger.info(f"Getting segmentation info from NIfTI file: {nii_path}")
    try:
        segmentation_nifti = nib.load(nii_path)
        header = segmentation_nifti.header

        affine = segmentation_nifti.affine
        voxel_size = header.get_zooms()
        data_type = header['datatype']
        description = header['descrip'].tobytes().decode('utf-8').strip()
        intent_code = header['intent_code']
        intent_name = header.get_intent()[0]

        info = {
            'Affine Matrix': affine,
            'Voxel Size': voxel_size,
            'Data Type': data_type,
            'Description': description,
            'Intent Code': intent_code,
            'Intent Name': intent_name
        }

        logger.info("Segmentation metadata extracted successfully.")
        return info
    except Exception as e:
        logger.error(f"Failed to extract segmentation info from '{nii_path}': {e}")
        raise e


def get_affine(image: sitk.Image):
    """
    Computes the affine transformation matrix for Nibabel from a SimpleITK image.

    Parameters:
    -----------
    image : sitk.Image
        The SimpleITK image.

    Returns:
    --------
    affine : np.ndarray
        The affine transformation matrix.
    """
    import numpy as np

    # Get the direction cosines, spacing, and origin from the SimpleITK image
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    # Construct the affine transformation matrix
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin

    # Convert from LPS (used by SimpleITK) to RAS (used by Nibabel)
    affine[0, :] *= -1
    affine[1, :] *= -1

    return affine




def save_segmentation(mask_image: sitk.Image, filename:str):
    """
    Saves the segmentation mask as a NIfTI file using Nibabel.

    Parameters:
    -----------
    mask_image : sitk.Image
        The segmentation mask image to save.
    filename : str
        The filename to save the NIfTI file as.
    """    

    mask = sitk.GetArrayFromImage(mask_image)

    mask = np.transpose(mask,(2, 1, 0))

    affine = get_affine(mask_image)

    nifti_image = nib.Nifti1Image(mask, affine)

    nib.save(nifti_image,filename)
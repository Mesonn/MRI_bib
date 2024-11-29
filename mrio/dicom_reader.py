# MRI_bib/mrio/dicom_reader.py

import os
import logging
import SimpleITK as sitk
import pydicom
import numpy as np
from utils.logger import setup_logger
from typing import Dict, Any

logger = setup_logger(__name__)

def read_dicom_series(dicom_folder):
    """
    Read a DICOM series from a folder.

    Parameters
    ----------
    dicom_folder : str
        Path to the folder containing DICOM files.

    Returns
    -------
    image : SimpleITK.Image
        The loaded DICOM image as a SimpleITK object.
    """
    logger.info(f"Reading DICOM series from folder: {dicom_folder}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    if not dicom_names:
        logger.error(f"No DICOM files found in folder: {dicom_folder}")
        raise FileNotFoundError(f"No DICOM files found in folder: {dicom_folder}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    logger.info("DICOM series read successfully.")
    return image

def read_multiple_echos_one_folder(dicom_folder):
    """
    Read multiple echo images from a single DICOM folder.

    Parameters
    ----------
    dicom_folder : str
        Path to the folder containing DICOM files for multiple echoes.

    Returns
    -------
    images : list of SimpleITK.Image
        List of images for each echo time.
    echo_times : list of float
        Corresponding echo times.
    """
    logger.info(f"Reading multiple echoes from folder: {dicom_folder}")
    dicom_files = [
        os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)
        if not f.startswith('.') and os.path.isfile(os.path.join(dicom_folder, f))
    ]
    echo_time_dict = {}
    for dcm_file in dicom_files:
        try:
            dicom_data = pydicom.dcmread(dcm_file)
            echo_time = float(dicom_data.EchoTime)
            instance_number = int(dicom_data.InstanceNumber)
            if echo_time not in echo_time_dict:
                echo_time_dict[echo_time] = []
            echo_time_dict[echo_time].append((instance_number, dcm_file))
        except Exception as e:
            logger.warning(f"Failed to read DICOM file {dcm_file}: {e}")
            continue
    images = []
    echo_times = sorted(echo_time_dict.keys())
    for echo_time in echo_times:
        files = echo_time_dict[echo_time]
        # Sort files by InstanceNumber
        files.sort(key=lambda x: x[0])
        sorted_files = [file_tuple[1] for file_tuple in files]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        image = reader.Execute()
        images.append(image)
        logger.info(f"Echo time {echo_time} ms: Read {len(sorted_files)} images.")
    logger.info(f"Total echoes read: {len(images)} with echo times: {echo_times}")
    return images, echo_times

def read_multiple_spin_lock_series(dicom_folders):
    """
    Read multiple spin-lock images from separate DICOM folders.

    Parameters
    ----------
    dicom_folders : list of str
        List of paths to folders containing DICOM files for each spin-lock time.

    Returns
    -------
    images : list of SimpleITK.Image
        List of images for each spin-lock time.
    tsl_values : list of float
        Corresponding spin-lock times.
    """
    logger.info("Reading multiple spin-lock series from separate folders.")
    images = []
    tsl_values = []
    for folder in dicom_folders:
        image = read_dicom_series(folder)
        images.append(image)
        # Extract TSL from folder name or DICOM metadata
        # Example assumes folder names contain TSL values like 'TSL_10ms'
        try:
            tsl_str = os.path.basename(folder).split('_')[-1]
            tsl = float(tsl_str.replace('ms', ''))
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract TSL from folder name '{folder}': {e}")
            tsl = np.nan
        tsl_values.append(tsl)
        logger.info(f"Folder '{folder}' read with TSL: {tsl} ms")
    # Sort images and TSLs based on TSL values, handling NaNs
    valid_indices = [i for i, tsl in enumerate(tsl_values) if not np.isnan(tsl)]
    sorted_indices = sorted(valid_indices, key=lambda i: tsl_values[i])
    images_sorted = [images[i] for i in sorted_indices]
    tsl_values_sorted = [tsl_values[i] for i in sorted_indices]
    logger.info(f"Sorted spin-lock images by TSL values: {tsl_values_sorted}")
    return images_sorted, tsl_values_sorted



def print_sitk_image_info(image: sitk.Image):
    """
    Prints basic spatial information and selected DICOM metadata from a SimpleITK.Image object.

    Parameters:
    ----------
    image : sitk.Image
        The SimpleITK Image object from which to extract information.
    """
    # Extract Spatial Information
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    pixel_type = image.GetPixelIDTypeAsString()
    dimension = image.GetDimension()

    # Print Spatial Information
    print("\n--- Spatial Information ---")
    print(f"Dimension: {dimension}")
    print(f"Size: {size}")
    print(f"Spacing: {spacing}")
    print(f"Origin: {origin}")
    print(f"Direction: {direction}")
    print(f"Pixel Type: {pixel_type}")

    # List of Common DICOM Tags to Extract
    common_dicom_tags = {
        'Patient Name': '0010|0010',
        'Patient ID': '0010|0020',
        'Study Date': '0008|0020',
        'Study Time': '0008|0030',
        'Modality': '0008|0060',
        'Manufacturer': '0008|0070',
        'Image Type': '0008|0008',
        'Slice Location': '0020|1041',
        'Echo Time': '0018|0081',
        'Repetition Time': '0018|0080'
    }

    # Extract and Print Selected DICOM Metadata
    print("\n--- Selected DICOM Metadata ---")
    for tag_name, tag_code in common_dicom_tags.items():
        try:
            value = image.GetMetaData(tag_code)
            print(f"{tag_name} ({tag_code}): {value}")
        except Exception as e:
            print(f"{tag_name} ({tag_code}): Not Available")


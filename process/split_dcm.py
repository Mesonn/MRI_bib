import os
from pathlib import Path
import logging
import SimpleITK as sitk 
import pydicom
import numpy as np 
from typing import Dict, Any, List, Tuple
from utils.logger import setup_logger


# Setting up the logger 
logger = setup_logger(__name__)


def split_dcm(dcm_list: list) -> List[List[str]]:
    """
    Splits a list of DICOM file paths into separate lists based on SliceLocation and echoes.

    Parameters
    ----------
    dcm_list : list
        List of DICOM file paths.

    Returns
    -------
    echo_list : list of lists
        Nested list where each sublist contains DICOM files for a specific echo across all slices.
    """

    logger.info("Starting split_dcm function")
    location: Dict[float, List[str]] = {}
    skipped_files = 0

    for f in dcm_list:
        try:
            d = pydicom.dcmread(f)
        except Exception as e:
            logger.warning(f"failed to read DICOM file {f}: {e}")
            skipped_files += 1 
            continue

        slice_loc = getattr(d, 'SliceLocation', None)
        if slice_loc is not None:
            slice_loc_value = slice_loc.value
            location.setdefault(slice_loc_value,[]).append(f)
            logger.debug(f"Added file {f} to slice location {slice_loc_value}.")
        else:
            logger.warning(f"SliceLocation not found in DICOM file {f}.")

        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} DICOM files due to read errors.")

        locations = check_locations(locations)
        split_dcmList = [locations[key] for key in sorted(locations.keys())]
        
        if not split_dcmList:
            logger.error("No valid DICOM files found after processing.")
            return []

        echo_count = len(split_dcmList[0])
        echo_list: List[List[str]] = [[] for _ in range(echo_count)]
        logger.info(f"Number of echoes per slice: {echo_count}")

        for echos in split_dcmList:
            for idx, echo in enumerate(echos):
                echo_list[idx].append(echo)
                logger.debug(f"Assigned file {echo} to echo index {idx}.")

        logger.info("Completed split_dcm function.")
        return echo_list



def check_locations(locations: Dict[float, List[str]]) -> Dict[float, List[str]]:
    """
    Validates and corrects slice locations to ensure consistent echo counts.

    Parameters
    ----------
    locations : dict
        Dictionary mapping SliceLocation to list of DICOM file paths.

    Returns
    -------
    locations : dict
        Corrected dictionary after merging inconsistent slice locations.
    """
    logger.info("Starting check_locations function.")
    keys = list(locations.keys())
    ls = [len(locations[key]) for key in keys]
    echos = np.median(ls)
    inconsistent_keys = [i for i, l in enumerate(ls) if (l - echos) != 0.0]

    if len(inconsistent_keys) == 2:
        key1, key2 = keys[inconsistent_keys[0]], keys[inconsistent_keys[1]]
        logger.warning(f"Inconsistent echo counts found at slice locations {key1} and {key2}. Merging these slices.")
        locations[key1].extend(locations[key2])
        del locations[key2]
    elif len(inconsistent_keys) > 0:
        logger.warning(f"More than two inconsistent slice locations found: {[keys[i] for i in inconsistent_keys]}. Manual review may be required.")

    logger.info("Completed check_locations function.")
    return locations


def split_dcm(dcm_list: list) -> Dict[float, List[str]]:
    """
    Splits a list of DICOM file paths into separate lists based on EchoTime.

    Parameters
    ----------
    dcm_list : list
        List of DICOM file paths.

    Returns
    -------
    echo_dict : dict
        Dictionary where each key is an echo time, and each value is a list of DICOM files for that echo time across all slices.
    """
    logger.info("Starting split_dcm function")
    echo_dict: Dict[float, List[Tuple[float, str]]] = {}
    skipped_files = 0

    for f in dcm_list:
        try:
            d = pydicom.dcmread(f)
        except Exception as e:
            logger.warning(f"Failed to read DICOM file {f}: {e}")
            skipped_files += 1 
            continue

        slice_loc = getattr(d, 'SliceLocation', None)
        echo_time = getattr(d, 'EchoTime', None)

        if slice_loc is not None and echo_time is not None:
            slice_loc_value = float(slice_loc)
            echo_time_value = float(echo_time)
            echo_dict.setdefault(echo_time_value, []).append((slice_loc_value, f))
            logger.debug(f"Added file {f} to echo time {echo_time_value} and slice location {slice_loc_value}.")
        else:
            logger.warning(f"SliceLocation or EchoTime not found in DICOM file {f}.")

    if skipped_files > 0:
        logger.info(f"Skipped {skipped_files} DICOM files due to read errors.")

    # Sort files within each echo time by slice location
    for echo_time in echo_dict:
        files = echo_dict[echo_time]
        files.sort(key=lambda x: x[0])  # Sort by slice location
        # Replace list of tuples with list of file paths
        echo_dict[echo_time] = [f for _, f in files]

    # Check for consistent number of slices across echoes
    slice_counts = [len(files) for files in echo_dict.values()]
    if len(set(slice_counts)) > 1:
        logger.warning(f"Inconsistent number of slices across echoes: {slice_counts}")
    else:
        logger.info(f"Number of slices per echo: {slice_counts[0]}")

    logger.info("Completed split_dcm function.")
    return echo_dict

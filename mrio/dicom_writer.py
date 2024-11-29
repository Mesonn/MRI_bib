import os
from pathlib import Path

import natsort
import nibabel as nib
import numpy as np
import pydicom

def mask_to_dicom(dcm_folder: Path, nii_file: Path, out_folder: Path):
    """
    Converts a NIfTI mask file to a series of DICOM files, aligning with the DICOM series in dcm_folder.

    Parameters:
    ----------
    dcm_folder : Path
        Path to the reference DICOM folder.
    nii_file : Path
        Path to the input NIfTI mask file.
    out_folder : Path
        Path to the output folder where converted DICOM mask files will be saved.
    """
    mask = np.transpose(np.array(nib.load(nii_file).dataobj), (1, 0, 2))
    dicom_files = natsort.natsorted([f for f in dcm_folder.glob("*.dcm")])
    mask = mask.astype("uint16")
    for i, dcm_file in enumerate(dicom_files):
        if i >= mask.shape[2]:
            break
        ds = pydicom.dcmread(dcm_file)
        ds.PixelData = mask[:, :, i].tobytes()
        ds.save_as(out_folder / dcm_file.name)

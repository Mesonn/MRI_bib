"""
mrio: Input/output operations for MRI data.
"""
from .dicom_writer import(
    mask_to_dicom
)

from .nifti_writer import(
    save_nifti_sitk,
    save_nifti_nibabel

)




from .dicom_reader import (
    read_dicom_series,
    read_multiple_echos_one_folder,
    read_multiple_spin_lock_series
)
from .nifti_reader import (
    load_segmentation_nii,
    load_segmentation_data,
    get_segmentation_info
)
# from .nifti_reader import (
#     read_nifti_file
# )

__all__ = [
    'read_dicom_series',
    'read_multiple_echos_one_folder',
    'read_multiple_spin_lock_series',
    'load_segmentation_nii',
    'load_segmentation_data',
    'get_segmentation_info',
    'mask_to_dicom',
    'save_nifti_sitk',
    'save_nifti_nibabel'

]

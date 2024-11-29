import tempfile
from pathlib import Path
from mrio import mask_to_dicom,save_nifti_sitk,save_nifti_nibabel
from process.split_dcm import split_dcm
import SimpleITK as sitk
import nibabel as nib
import numpy as np


def transform(
    input_dicom_folder_1: Path,
    input_mask_file: Path,
    input_dicom_folder_2: Path,
    out_nii_file: Path,
    reverse: bool = False,
):
    """
    Transforms the mask image to align with the images in the second DICOM folder, using the images in the first DICOM folder as reference.
    The resulting image is saved in NIFTI format.

    Parameters:
    input_dicom_folder_1 (Path): Path to the first DICOM folder.
    input_mask_file (Path): Path to the mask file.
    input_dicom_folder_2 (Path): Path to the second DICOM folder.
    out_nii_file (Path): Path to the output NIFTI file.
    reverse (bool, optional): Flag to specify whether the images in the second DICOM folder should be read in reverse order. Default is False.
    """

    # Initialize a SimpleITK ImageSeriesReader and ImageFileWriter
    reader = sitk.ImageSeriesReader()
    writer = sitk.ImageFileWriter()

    # Set reverse to False if it is None
    reverse = False if reverse is None else reverse

    while True:
        # Read the images from input_dicom_folder_2 using the ImageSeriesReader
        dicom_files = reader.GetGDCMSeriesFileNames(input_dicom_folder_2.as_posix())
        # Use the updated split_dcm function
        dicom_dict = split_dcm(dicom_files)
        if not dicom_dict:
            raise ValueError("No DICOM files found in the specified folder.")

        # Select the first echo time
        first_echo_time = sorted(dicom_dict.keys())[0]
        dicom_names = dicom_dict[first_echo_time]

        # Reverse the order of the images if reverse is True
        if reverse:
            dicom_names = dicom_names[::-1]
        reader.SetFileNames(dicom_names)
        target = reader.Execute()

        # Create a temporary directory and convert the mask file to a DICOM image using the mask_to_dicom function
        temp_dir_mask_as_dcm = tempfile.TemporaryDirectory()
        mask_to_dicom(
            input_dicom_folder_1, input_mask_file, Path(temp_dir_mask_as_dcm.name)
        )

        # Read the mask image from the temporary directory
        mask_dicom_files = reader.GetGDCMSeriesFileNames(temp_dir_mask_as_dcm.name)
        # Use the updated split_dcm function for the mask DICOM files
        mask_dicom_dict = split_dcm(mask_dicom_files)
        if not mask_dicom_dict:
            raise ValueError("No DICOM files found in the mask temporary directory.")

        # Select the first echo time for the mask
        mask_echo_time = sorted(mask_dicom_dict.keys())[0]
        mask_dicom_names = mask_dicom_dict[mask_echo_time]

        reader.SetFileNames(mask_dicom_names)
        mask = reader.Execute()

        # Cast the mask and target images to sitkFloat32 to prepare them for registration
        mask = sitk.Cast(mask, sitk.sitkFloat32)
        target = sitk.Cast(target, sitk.sitkFloat32)

        # Registration of mask_as_dcm to input_dicom_folder_2
        resampleFilter = sitk.ResampleImageFilter()
        # Use nearest neighbor interpolation
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)

        # Transform mask_image
        resampleFilter.SetSize(target.GetSize())
        resampleFilter.SetOutputOrigin(target.GetOrigin())
        resampleFilter.SetOutputSpacing(target.GetSpacing())
        resampleFilter.SetOutputDirection(target.GetDirection())
        resampleFilter.SetOutputPixelType(sitk.sitkInt8)
        resampleFilter.SetDefaultPixelValue(0.0)

        # Generate registered image
        registeredImg = resampleFilter.Execute(mask)
        registeredImg = sitk.Cast(registeredImg, sitk.sitkUInt8)
        writer.SetFileName(out_nii_file.as_posix())

        writer.Execute(registeredImg)
        img_nifti = nib.load(out_nii_file)
        img = img_nifti.get_fdata()
        if reverse:
            break
        if img.max() == 0:
            reverse = True
        else:
            break

    # Save registered mask as NIFTI
    img_nifti = nib.Nifti1Image(img, img_nifti.affine, img_nifti.header)
    nib.save(img_nifti, out_nii_file)
    temp_dir_mask_as_dcm.cleanup()



def transform_mask(
    input_dicom_folder_1: Path,
    input_mask_file: Path,
    input_dicom_folder_2: Path,
    reverse: bool = False,
    save_to_disk: bool = False,
    save_method: str = 'sitk',  # 'sitk' or 'nibabel'
) -> sitk.Image:
    """
    Transforms the mask image to align with the images in the second DICOM folder,
    using the images in the first DICOM folder as reference.
    Optionally saves the resulting image in NIfTI (.nii.gz) format using the specified method.

    The saved filename format is "Transformed_mask_<input_dicom_folder_2.name>.nii.gz".

    Parameters:
    input_dicom_folder_1 (Path): Path to the first DICOM folder (reference).
    input_mask_file (Path): Path to the mask file.
    input_dicom_folder_2 (Path): Path to the second DICOM folder (target).
    reverse (bool, optional): Flag to specify whether the images in the second DICOM folder should be read in reverse order.
                               Useful if the slice order might be inverted. Default is False.
    save_to_disk (bool, optional): Whether to save the resulting NIFTI image to disk. Default is False.
    save_method (str, optional): The method to use for saving the NIFTI file. Options are 'sitk' or 'nibabel'.
                                 Default is 'sitk'.

    Returns:
    sitk.Image: The transformed and registered mask as a SimpleITK Image.

    Raises:
    ValueError: If no DICOM files are found in the specified folders or if an invalid save_method is provided.
    IOError: If saving the NIfTI image fails.
    """
    # Initialize a SimpleITK ImageSeriesReader
    reader = sitk.ImageSeriesReader()

    # Ensure 'reverse' is a boolean
    reverse = False if reverse is None else reverse

    # Validate save_method
    if save_method not in ['sitk', 'nibabel']:
        raise ValueError("save_method must be either 'sitk' or 'nibabel'.")

    # Main processing loop: Attempt registration, possibly reversing slice order if needed
    while True:
        # Read the DICOM series from input_dicom_folder_2
        dicom_files = reader.GetGDCMSeriesFileNames(str(input_dicom_folder_2))
        dicom_dict = split_dcm(dicom_files)
        if not dicom_dict:
            raise ValueError(f"No DICOM files found in the target folder: {input_dicom_folder_2}")

        # Select the first echo time
        first_echo_time = sorted(dicom_dict.keys())[0]
        dicom_names = dicom_dict[first_echo_time]

        # Optionally reverse the order of DICOM slices
        if reverse:
            dicom_names = dicom_names[::-1]
        reader.SetFileNames(dicom_names)
        target = reader.Execute()

        # Convert the mask file to DICOM format in a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir_mask_as_dcm_name:
            temp_dir_mask_as_dcm = Path(temp_dir_mask_as_dcm_name)
            mask_to_dicom(input_dicom_folder_1, input_mask_file, temp_dir_mask_as_dcm)

            # Read the mask DICOM series
            mask_dicom_files = reader.GetGDCMSeriesFileNames(str(temp_dir_mask_as_dcm))
            mask_dicom_dict = split_dcm(mask_dicom_files)
            if not mask_dicom_dict:
                raise ValueError(f"No DICOM files found in the mask temporary directory: {temp_dir_mask_as_dcm}")

            # Select the first echo time for the mask
            mask_echo_time = sorted(mask_dicom_dict.keys())[0]
            mask_dicom_names = mask_dicom_dict[mask_echo_time]

            reader.SetFileNames(mask_dicom_names)
            mask = reader.Execute()

        # Cast images to Float32 for processing
        mask = sitk.Cast(mask, sitk.sitkFloat32)
        target = sitk.Cast(target, sitk.sitkFloat32)

        # Set up the resampling filter
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampleFilter.SetSize(target.GetSize())
        resampleFilter.SetOutputOrigin(target.GetOrigin())
        resampleFilter.SetOutputSpacing(target.GetSpacing())
        resampleFilter.SetOutputDirection(target.GetDirection())
        resampleFilter.SetOutputPixelType(sitk.sitkInt8)
        resampleFilter.SetDefaultPixelValue(0.0)

        # Generate the registered image
        registeredImg = resampleFilter.Execute(mask)
        registeredImg = sitk.Cast(registeredImg, sitk.sitkUInt8)

        # Extract the NumPy array for verification
        img_array = sitk.GetArrayFromImage(registeredImg)  # Shape: [z, y, x]

        # Check if the registration was successful
        if reverse:
            # If already reversed once, do not attempt again
            break
        if img_array.max() == 0:
            # If the mask is empty, attempt with reversed slice order
            reverse = True
        else:
            # Successful registration
            break

    # Optionally save the registered image to disk
    if save_to_disk:
        try:
            if save_method == 'sitk':
                save_nifti_sitk(registeredImg, input_dicom_folder_2)
            elif save_method == 'nibabel':
                save_nifti_nibabel(registeredImg, input_dicom_folder_2)
        except Exception as e:
            raise IOError(f"Failed to save NIfTI image using {save_method}: {e}")

    return registeredImg

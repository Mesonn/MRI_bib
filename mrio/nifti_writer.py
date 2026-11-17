import SimpleITK as sitk
from pathlib import Path
import nibabel as nib
import numpy as np
from .nifti_reader import get_affine


def save_nifti_sitk(image: sitk.Image, target_folder: Path):
    """
    Saves a SimpleITK image as a NIfTI (.nii.gz) file using SimpleITK's ImageFileWriter.

    The filename format is "Transformed_mask_<target_folder_name>.nii.gz".

    Parameters:
    image (sitk.Image): The SimpleITK image to be saved.
    target_folder (Path): The Path object of the target DICOM folder, used to derive the filename.

    Raises:
    IOError: If the image cannot be saved due to I/O issues.
    """
    # Extract the target folder name
    target_folder_name = target_folder.name

    # Construct the output filename
    output_filename = f"Transformed_mask_{target_folder_name}.nii.gz"
    output_path = Path(output_filename)

    # Initialize the SimpleITK ImageFileWriter
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_path))

    # Execute the write operation
    try:
        writer.Execute(image)
        print(f"Image successfully saved to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save NIfTI image with SimpleITK: {e}")




def save_nifti_nibabel(image: sitk.Image, target_folder: Path):
    """
    Saves a SimpleITK image as a NIfTI (.nii.gz) file using NiBabel.

    The filename format is "Transformed_mask_<target_folder_name>.nii.gz".

    Parameters:
    image (sitk.Image): The SimpleITK image to be saved.
    target_folder (Path): The Path object of the target DICOM folder, used to derive the filename.

    Raises:
    IOError: If the image cannot be saved due to I/O issues.
    """
    # Extract the target folder name
    target_folder_name = target_folder.name

    # Construct the output filename
    output_filename = f"Transformed_mask_{target_folder_name}.nii.gz"
    output_path = Path(output_filename)

    # SimpleITK arrays are [z, y, x] with an LPS geometry; NIfTI/NiBabel expects [x, y, z]
    # with an RAS affine.
    img_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    affine = get_affine(image)

    # Create a Nifti1Image with NiBabel
    nifti_img = nib.Nifti1Image(img_array, affine)

    # Execute the save operation
    try:
        nib.save(nifti_img, str(output_path))
        print(f"Image successfully saved to {output_path} using NiBabel")
    except Exception as e:
        raise IOError(f"Failed to save NIfTI image with NiBabel: {e}")

import SimpleITK as sitk
import numpy as np
import os

def merge_masks_select_rois(mask1_path, mask2_path, roi_sequence, output_path):
    """
    Merge two mask images by selecting specific ROIs from each mask in a specified sequence.

    Parameters:
    - mask1_path (str): Path to the first mask image.
    - mask2_path (str): Path to the second mask image.
    - roi_sequence (list of tuples): List specifying the order of ROIs to include in the final mask.
                                     Each tuple is (mask_number, roi_label), where mask_number is 1 or 2.
                                     Example: [(1, 1), (2, 2), (1, 3)]
    - output_path (str): Path to save the merged mask.

    Returns:
    - None
    """

    # Validate input mask paths
    if not os.path.exists(mask1_path):
        raise FileNotFoundError(f"Mask1 not found at path: {mask1_path}")
    if not os.path.exists(mask2_path):
        raise FileNotFoundError(f"Mask2 not found at path: {mask2_path}")

    # Read the mask images
    mask1 = sitk.ReadImage(mask1_path)
    mask2 = sitk.ReadImage(mask2_path)

    # Ensure both masks have the same size, spacing, and origin
    if mask1.GetSize() != mask2.GetSize():
        raise ValueError("Mask1 and Mask2 have different dimensions.")
    if mask1.GetSpacing() != mask2.GetSpacing():
        print(mask1.GetSpacing(), mask2.GetSpacing())
        raise ValueError("Mask1 and Mask2 have different spacings.")
    if mask1.GetOrigin() != mask2.GetOrigin():
        raise ValueError("Mask1 and Mask2 have different origins.")
    if mask1.GetDirection() != mask2.GetDirection():
        raise ValueError("Mask1 and Mask2 have different directions.")

    # Convert masks to NumPy arrays for easier manipulation
    mask1_array = sitk.GetArrayFromImage(mask1)
    mask2_array = sitk.GetArrayFromImage(mask2)

    # Initialize the final mask array with zeros (background)
    final_mask_array = np.zeros_like(mask1_array, dtype=np.uint8)

    # Initialize label counter for the final mask
    label_counter = 1

    # Iterate through the ROI sequence and assign labels accordingly
    for entry in roi_sequence:
        mask_number, roi_label = entry

        if mask_number == 1:
            current_mask = mask1_array
        elif mask_number == 2:
            current_mask = mask2_array
        else:
            raise ValueError(f"Invalid mask number {mask_number}. Only 1 or 2 are allowed.")

        # Create a binary mask for the current ROI
        roi_binary = (current_mask == roi_label)

        if not np.any(roi_binary):
            print(f"Warning: ROI label {roi_label} not found in Mask {mask_number}. Skipping.")
            continue

        # Assign the current label to the final mask where the ROI is present
        final_mask_array[roi_binary] = label_counter
        print(f"Assigned label {label_counter} to ROI {roi_label} from Mask {mask_number}.")

        label_counter += 1

    # Convert the final mask array back to a SimpleITK image
    final_mask = sitk.GetImageFromArray(final_mask_array)
    final_mask.CopyInformation(mask1)  # Preserve spatial metadata

    # Save the final merged mask
    sitk.WriteImage(final_mask, output_path)
    print(f"Merged mask saved to {output_path}.")

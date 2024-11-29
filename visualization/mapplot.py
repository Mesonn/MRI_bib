# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from typing import List, Optional

from utils.logger import setup_logger
from models.roi import ROI

# Initialize logger
logger = setup_logger(__name__)


def show_slice(
    image: sitk.Image,
    slice_index: int,
    title: str = "Slice",
    cmap: str = 'gray'
):
    """
    Display a single slice of a 3D MRI image.

    Parameters
    ----------
    image : SimpleITK.Image
        The 3D MRI image to display.
    slice_index : int
        The index of the slice to display along the Z-axis.
    title : str, optional
        Title of the plot (default is "Slice").
    cmap : str, optional
        Colormap to use for the image (default is 'gray').
    """
    logger.info(f"Displaying slice {slice_index} of the image.")
    image_array = sitk.GetArrayFromImage(image)

    if slice_index < 0 or slice_index >= image_array.shape[0]:
        logger.error(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")
        raise IndexError(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_array[slice_index, :, :], cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_mask(
    image: sitk.Image,
    rois: Optional[List[ROI]] = None,
    slice_index: int = 0,
    title: str = "Mask Overlay",
    alpha: float = 0.5
):
    """
    Display a specific slice of multiple ROIs overlaid on the MRI image, each with its respective label ID.

    Parameters
    ----------
    image : SimpleITK.Image
        The base MRI image.
    rois : Optional[List[ROI]]
        List of ROI instances to overlay.
    slice_index : int, optional
        The index of the slice to display along the Z-axis (default is 0).
    title : str, optional
        Title of the plot (default is "Mask Overlay").
    alpha : float, optional
        Transparency level for the mask overlay (default is 0.5).
    """
    logger.info(f"Displaying mask overlay on slice {slice_index}.")

    image_array = sitk.GetArrayFromImage(image)

    if slice_index < 0 or slice_index >= image_array.shape[0]:
        logger.error(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")
        raise IndexError(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_array[slice_index, :, :], cmap='gray')

    if rois:
        label_map = np.zeros_like(image_array[slice_index, :, :], dtype=np.int32)
        label_ids = []

        for roi in rois:
            if roi.mask.shape != image_array.shape:
                logger.error(f"ROI Label {roi.label_id} mask shape {roi.mask.shape} does not match image shape {image_array.shape}.")
                raise ValueError(f"ROI Label {roi.label_id} mask shape {roi.mask.shape} does not match image shape {image_array.shape}.")

            mask_slice = roi.mask[slice_index, :, :]
            current_label = roi.label_id
            label_map[mask_slice > 0] = current_label
            label_ids.append(current_label)

        if label_ids:
            unique_labels = sorted(list(set(label_ids)))
            # Define colors for each label. 'tab10' has 10 distinct colors; adjust if more labels are present.
            base_cmap = plt.get_cmap('tab10')
            colors = [(0, 0, 0, 0)] + [base_cmap(i % 10) for i in range(len(unique_labels))]  # Background transparent
            custom_cmap = ListedColormap(colors)

            # Plot the label_map
            plt.imshow(label_map, cmap=custom_cmap, alpha=alpha, vmin=0, vmax=len(unique_labels))

            # Create legend handles
            handles = [mpatches.Patch(color='none', label='Background')]
            for idx, label_id in enumerate(unique_labels, start=1):
                color = custom_cmap(idx)
                label = f'Label {label_id}'
                handles.append(mpatches.Patch(color=color, label=label))

            plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.title(title)
    plt.axis('off')
    plt.show()




def show_rois_with_mean(
    image: sitk.Image,
    rois: Optional[List[ROI]],
    slice_index: int = 0,
    title: str = "ROIs with Mean Parameter",
    cmap: str = 'jet',
    alpha: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Show the mean parameter values for each ROI overlaid on the MRI image,
    using a single colormap that maps mean values to colors.

    Parameters
    ----------
    image : SimpleITK.Image
        The base MRI image.
    rois : Optional[List[ROI]]
        List of ROI instances to overlay.
    slice_index : int, optional
        The index of the slice to display along the Z-axis (default is 0).
    title : str, optional
        Title of the plot (default is "ROIs with Mean Parameter").
    cmap : str, optional
        Colormap to represent mean parameter values (default is 'jet').
    alpha : float, optional
        Transparency level for the mean map overlay (default is 0.5).
    save_path : Optional[str], optional
        File path to save the visualization (default is None).
    """
    if save_path:
        display = False
        logger.info(f"Display ROIs with mean parameter disabled; saving to {save_path}.")
    else:
        display = True

    logger.info(f"Displaying ROIs with mean parameter on slice {slice_index}.")

    image_array = sitk.GetArrayFromImage(image)

    if slice_index < 0 or slice_index >= image_array.shape[0]:
        logger.error(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")
        raise IndexError(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")

    # Collect mean values to determine vmin and vmax for colormap
    mean_values = [roi.mean for roi in rois if roi.mean is not None]
    if not mean_values:
        logger.error("No valid mean values found in the provided ROIs.")
        raise ValueError("At least one ROI with a valid mean value must be provided.")
    vmin = min(mean_values)
    vmax = max(mean_values)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_array[slice_index, :, :], cmap='gray')

    combined_mean_slice = np.full_like(image_array[slice_index, :, :], np.nan, dtype=np.float32)

    for roi in rois:
        if roi.mean is not None and roi.mask.shape == image_array.shape:
            mask_slice = roi.mask[slice_index, :, :] > 0
            combined_mean_slice[mask_slice] = roi.mean
        else:
            logger.warning(f"ROI Label {roi.label_id} has no mean value or mask shape mismatch.")

    # Mask the combined mean slice
    masked_mean_slice = np.ma.array(combined_mean_slice, mask=np.isnan(combined_mean_slice))

    # Display the combined mean map
    mean_im = plt.imshow(masked_mean_slice, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mean_im, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Parameter Value')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"ROIs with mean parameter saved to {save_path}.")

    if display:
        plt.show()


def show_parameter_map_with_rois(
    image: sitk.Image,
    rois: Optional[List[ROI]],
    slice_index: int = 0,
    title: str = "Parameter Map with ROIs",
    cmap_param: str = 'jet',
    alpha_param: float = 0.6,
    cmap_roi: str = 'autumn',
    alpha_roi: float = 0.1,
    save_path: Optional[str] = None
):
    """
    Show the parameter map overlaid on the MRI image with all ROIs highlighted.

    Parameters
    ----------
    image : SimpleITK.Image
        The base MRI image.
    rois : Optional[List[ROI]]
        List of ROI instances to overlay.
    slice_index : int, optional
        The index of the slice to display along the Z-axis (default is 0).
    title : str, optional
        Title of the plot (default is "Parameter Map with ROIs").
    cmap_param : str, optional
        Colormap for the parameter map (default is 'jet').
    alpha_param : float, optional
        Transparency level for the parameter map overlay (default is 0.6).
    cmap_roi : str, optional
        Colormap for the ROI masks overlay (default is 'autumn').
    alpha_roi : float, optional
        Transparency level for the ROI masks overlay (default is 0.3).
    save_path : Optional[str], optional
        File path to save the visualization (default is None).
    """
    if save_path:
        display = False  # Will save instead of displaying
        logger.info(f"Display parameter map with ROIs disabled; saving to {save_path}.")
    else:
        display = True

    logger.info(f"Displaying parameter map with ROIs on slice {slice_index}.")

    image_array = sitk.GetArrayFromImage(image)

    if slice_index < 0 or slice_index >= image_array.shape[0]:
        logger.error(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")
        raise IndexError(f"Slice index {slice_index} is out of bounds for image with shape {image_array.shape}.")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_array[slice_index, :, :], cmap='gray')

    # Initialize combined_param_map with NaNs
    combined_param_map = np.full_like(image_array, np.nan, dtype=np.float32)

    for roi in rois:
        if roi.pmap is not None and roi.mask.shape == image_array.shape:
            combined_param_map[roi.mask > 0] = roi.pmap[roi.mask > 0]
        else:
            logger.warning(f"ROI Label {roi.label_id} has no parameter map or mask shape mismatch.")

    if np.any(~np.isnan(combined_param_map)):
        param_slice = combined_param_map[slice_index, :, :]
        param_im = plt.imshow(param_slice, cmap=cmap_param, alpha=alpha_param)
        cbar = plt.colorbar(param_im, fraction=0.046, pad=0.04)
        cbar.set_label('Parameter Value')
    else:
        logger.error("No valid parameter maps found in the provided ROIs.")
        raise ValueError("At least one ROI with a valid parameter map must be provided.")

    # Overlay the combined ROI masks
    combined_mask = np.zeros_like(image_array[slice_index, :, :], dtype=bool)

    for roi in rois:
        if roi.mask.shape != image_array.shape:
            logger.error(f"ROI Label {roi.label_id} mask shape {roi.mask.shape} does not match image shape {image_array.shape}.")
            raise ValueError(f"ROI Label {roi.label_id} mask shape {roi.mask.shape} does not match image shape {image_array.shape}.")
        combined_mask |= roi.mask[slice_index, :, :] > 0

    if np.any(combined_mask):
        plt.imshow(np.ma.masked_where(~combined_mask, combined_mask), cmap=cmap_roi, alpha=alpha_roi)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Parameter map with ROIs saved to {save_path}.")

    if display:
        plt.show()

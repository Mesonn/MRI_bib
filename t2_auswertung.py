import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from mrio import dicom_reader
from mrio import nifti_reader
from models.roi import create_rois, ROI  # Import create_rois and ROI class
from fitting.relaxation_fitting import RelaxationFittingModel
from process import transform,transform_mask,SignalFilter,merge_masks_select_rois
from fitting.analysis import compute_roi_statistics
from visualization.mapplot import (
    show_slice,
    show_mask,
    show_parameter_map_with_rois,
    show_rois_with_mean,
)

input_dicom_folder_T2 = '/home/gizmoo/Desktop/auswertungen/t2/19_T2Map_2D(Images)_00011'
save_folder = '/home/gizmoo/Desktop/auswertungen/t2/results'
seg_path = '/home/gizmoo/Desktop/auswertungen/t2/t2_mask.nii.gz'


images, echos = dicom_reader.read_multiple_echos_one_folder(input_dicom_folder_T2)
print("Echo Times:", echos)
print(f"Image Size: {images[0].GetSize()}")


relaxation = 'T2'
# Define fitting modell
fitting_model = RelaxationFittingModel(time_values= echos[1:],model_type=relaxation)
seg = nifti_reader.load_segmentation_nii(seg_path)
print(f"Segmentation Size: {seg.GetSize()}")

label_values = list(range(1,5))
rois = create_rois(seg, images[0], label_values)
print(f"Number of ROIs Created: {len(rois)}")


slice_index = 0  # Adjust based on your data
show_mask(
    image=images[0],
    rois=rois,
    slice_index=slice_index,
    title=f"Mask Overlay - Slice {slice_index}",
    alpha=0.5,
)

# Fit T2 curve for each ROI
for roi in rois:
    print(f"Fitting T2 curve for Label {roi.label_id}...")
    roi.fit(images=images[1:],fitting_model= fitting_model)
    roi.compute_statistics()
    print(f"Label {roi.label_id} - Mean T2: {roi.mean:.2f} ms, Std T2: {roi.std:.2f} ms")


# Visualize the combined T2 Parameter Map with all ROIs
show_parameter_map_with_rois(
    image=images[0],
    rois=rois,
    slice_index=slice_index,  # Adjust based on your data
    title=f"Combined T2 Parameter Map with ROIs - Slice {slice_index}",
    cmap_param='jet',       # Choose an appropriate colormap
    alpha_param=0.8,       # Adjust transparency as needed
    cmap_roi='autumn',     # Choose an appropriate colormap for ROIs
    vmin= 10,
    vmax= 60,
    alpha_roi=0.3,         # Adjust transparency as needed
    save_path= save_folder
)    
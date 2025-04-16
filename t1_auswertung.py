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




seg_path = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/t1_low_TR_1slice.nii.gz'

#TI folders
sl1 = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/39_t1_2D_TI25ms_BW977_HR_peFH_00236'
sl2 = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/38_t1_2D_TI50ms_BW977_HR_peFH_00161'
sl3 = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/40_t1_2D_TI100ms_BW977_HR_peFH_00006'
sl4 = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/41_t1_2D_TI500ms_BW977_HR_peFH_00081'
sl5 = '/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/42_t1_2D_TI1000ms_BW977_HR_peFH_00156'
# sl6 = '/home/gizmoo/Desktop/Dixon_Test/t1_high_TR/44_t12D_TI2000ms_TRhigh_BW977_HR_peFH_00343'
# sl7 = '/home/gizmoo/Desktop/Dixon_Test/t1_high_TR/43_t12D_TI3000ms_TRhigh_BW977_HR_peFH_00268'


folders = [sl1,sl2,sl3,sl4,sl5]
save_folder = "/home/gizmoo/Desktop/Dixon_Test/t1_small_TR/results"

images, _ = dicom_reader.read_multiple_spin_lock_series(folders)
TI = [25,50,100,500,1000]


print("inversion Times:", TI)
print(f"Image Size: {images[0].GetSize()}")

relaxation = 'T1'
# Define fitting modell
fitting_model = RelaxationFittingModel(time_values= TI,model_type=relaxation)
seg = nifti_reader.load_segmentation_nii(seg_path)
print(f"Segmentation Size: {seg.GetSize()}")



label_values = list(range(1,4))
rois = create_rois(seg, images[0], label_values)
print(f"Number of ROIs Created: {len(rois)}")

reference_image = images[-1]  # Use the last image
slice_index = 17  
show_mask(
    image=reference_image,
    rois=rois,
    slice_index=slice_index,
    title=f"Mask Overlay - Slice {slice_index}",
    alpha=0.5,
)

# Fit T2 curve for each ROI
for roi in rois:
    print(f"Fitting T1 curve for Label {roi.label_id}...")
    roi.fit(images=images,fitting_model= fitting_model)
    roi.compute_statistics()
    print(f"Label {roi.label_id} - Mean T1: {roi.mean:.2f} ms, Std T1: {roi.std:.2f} ms")


# Visualize the combined T2 Parameter Map with all ROIs
show_parameter_map_with_rois(
    image=images[0],
    rois=rois,
    slice_index=slice_index,  # Adjust based on your data
    title=f"Combined T2 Parameter Map with ROIs - Slice {slice_index}",
    cmap_param='jet',       # Choose an appropriate colormap
    alpha_param=0.8,       # Adjust transparency as needed
    cmap_roi='autumn',     # Choose an appropriate colormap for ROIs
    vmin= 100,
    vmax= 500,
    alpha_roi=0.3,         # Adjust transparency as needed
    save_path= save_folder
)    
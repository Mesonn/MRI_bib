#!/usr/bin/env python
"""
T1 Mapping Analysis Script for MRI Data
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from mrio import dicom_reader
from mrio import nifti_reader
from models.roi import create_rois, ROI
from fitting.relaxation_fitting import RelaxationFittingModel
from visualization.mapplot import (
    show_slice,
    show_mask,
    show_parameter_map_with_rois,
    show_rois_with_mean,
)




# Base directory and paths
base_dir = '/home/gizmoo/Desktop/t1_rho'
seg_path = f'{base_dir}/t1_rho.nii.gz'
save_folder = f"{base_dir}/results_new"




# Dictionary mapping TI values to their folder paths
ti_folders = {
    10: f'{base_dir}/2_T1rho_cor_NATURE_8contrasts_50214',
    40: f'{base_dir}/3_T1rho_cor_NATURE_8contrasts_50214',
    70: f'{base_dir}/4_T1rho_cor_NATURE_8contrasts_50214',
    100: f'{base_dir}/5_T1rho_cor_NATURE_8contrasts_50214',
    130: f'{base_dir}/6_T1rho_cor_NATURE_8contrasts_50214',
    160: f'{base_dir}/7_T1rho_cor_NATURE_8contrasts_50214'
    # 3000: f'{base_dir}/43_t12D_TI3000ms_TRhigh_BW977_HR_peFH_00268'
}

# Sort TI values and create ordered lists for both TI and folders
TI = sorted(ti_folders.keys())
folders = [ti_folders[ti] for ti in TI]

# Load images in proper order
print("Loading DICOM images...")
images, metadata = dicom_reader.read_multiple_spin_lock_series(folders)

# Verify loading order
print("Verifying image loading order:")
for i, (ti, folder) in enumerate(zip(TI, folders)):
    print(f"Image {i}: TI={ti}ms, Folder={Path(folder).name}")

print(f"Image Size: {images[0].GetSize()}")

# Use the highest TI image for better visualization/reference
reference_image = images[-1] 

# Configure fitting model
relaxation = 'T2'
fitting_model = RelaxationFittingModel(time_values=TI, model_type=relaxation)

# Load segmentation
seg = nifti_reader.load_segmentation_nii(seg_path)
print(f"Segmentation Size: {seg.GetSize()}")

# Create ROIs using the reference image (best anatomical detail)
label_values = list(range(1, 4))
rois = create_rois(seg, reference_image, label_values)
print(f"Number of ROIs Created: {len(rois)}")

# Choose slice to visualize
slice_index = 11

# Show mask overlay using the reference image
show_mask(
    image=reference_image,
    rois=rois,
    slice_index=slice_index,
    title=f"Mask Overlay - Slice {slice_index}",
    alpha=0.5,
)

# Plot sample T1 recovery curve (optional but helpful for verification)
sample_x, sample_y = 120, 120
signal_intensities = []
for img in images:
    img_array = sitk.GetArrayFromImage(img)
    signal_intensities.append(img_array[slice_index, sample_y, sample_x])

plt.figure(figsize=(8, 5))
plt.plot(TI, signal_intensities, 'o-', label='Signal Intensity')
plt.xlabel('Inversion Time (ms)')
plt.ylabel('Signal Intensity')
plt.title('T1 Recovery Curve at Sample Point')
plt.grid(True)
plt.savefig(f"{save_folder}/T1_sample_curve.png")
plt.close()

# Fit T1 curve for each ROI
for roi in rois:
    print(f"Fitting T1 curve for Label {roi.label_id}...")
    roi.fit(images=images, fitting_model=fitting_model)
    roi.compute_statistics()
    print(f"Label {roi.label_id} - Mean T1: {roi.mean:.2f} ms, Std T1: {roi.std:.2f} ms")

# Visualize the T1 parameter map with ROIs
show_parameter_map_with_rois(
    image=reference_image,
    rois=rois,
    slice_index=slice_index,
    title=f"T1 Parameter Map with ROIs - Slice {slice_index}",
    cmap_param='jet',
    alpha_param=0.8,
    cmap_roi='autumn',
    vmin=300,
    vmax=1000,
    alpha_roi=0.3,
    save_path=f"{save_folder}/T1_parameter_map.png"
)

# Also save ROIs with mean values for easy interpretation
show_rois_with_mean(
    image=reference_image,
    rois=rois,
    slice_index=slice_index,
    title=f"T1 Values by ROI - Slice {slice_index}",
    save_path=f"{save_folder}/T1_roi_means.png"
)


roi1 = rois[0]
voxels = [0,10,20]
for voxel in voxels:
    voxel_signal = roi1.signal[voxel, :] 
    fitting_model.plot_voxel_fit(ydata=voxel_signal, voxel_index=voxel)

for roi in rois:
    roi.plot_distribution()
print("T1 Mapping analysis complete!")
"""
PyRadiomics Feature Extraction + Interactive Visualization
Author: doffi
Year: 2025
Description:
  Extracts radiomics features and interactively displays
  each case (one at a time) with mask overlay.
"""

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor

# --- Paths ---
dataset_root = r"E:\Projects\UAB-Decision-Making\dataset\sample\CT"
image_folder = os.path.join(dataset_root, "image")
mask_folder  = os.path.join(dataset_root, "nodule_mask")

# --- PyRadiomics setup ---
extractor = featureextractor.RadiomicsFeatureExtractor()

# --- Loop through cases ---
for image_file in sorted(os.listdir(image_folder)):
    if not image_file.endswith(".nii.gz"):
        continue

    base_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
    mask_candidates = [m for m in os.listdir(mask_folder) if m.startswith(base_name)]
    if not mask_candidates:
        print(f"No mask found for {base_name}, skipping.")
        continue

    mask_file = os.path.join(mask_folder, mask_candidates[0])
    image_path = os.path.join(image_folder, image_file)

    # --- Read image & mask ---
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_file)
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)

    # --- Pick middle slice of mask ---
    z_indices = np.where(mask_np.sum(axis=(1,2)) > 0)[0]
    if len(z_indices) == 0:
        print(f" Empty mask for {base_name}, skipping visualization.")
        continue
    mid_slice = z_indices[len(z_indices)//2]

    # --- Visualization ---
    plt.figure(figsize=(6,6))
    plt.imshow(image_np[mid_slice], cmap='gray')
    plt.contour(mask_np[mid_slice], colors='r', linewidths=1)
    plt.title(f"{base_name} (slice {mid_slice})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()   # <== 👈 这句非常关键，会停下来等你看完窗口

    # --- Extract features ---
    print(f" Extracting features for {base_name} ...")
    result = extractor.execute(image, mask)
    print(f" {base_name}: extracted {len(result)} features.")

print("All cases processed successfully!")

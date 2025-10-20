"""
PyRadiomics Batch Feature Extraction + Visualization (自动保存特征 + label列)
Author: doffi
Year: 2025
"""

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radiomics import featureextractor
import sys

# === STEP 1: Set project paths ===
# Support command-line dataset path
if len(sys.argv) > 1:
    dataset_root = os.path.abspath(sys.argv[1])
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_root = os.path.join(project_root, "dataset", "sample", "CT")

print(f"Feature extraction dataset root: {dataset_root}")
image_folder = os.path.join(dataset_root, "image")
mask_folder = os.path.join(dataset_root, "nodule_mask")


# === STEP 2: Initialize PyRadiomics extractor ===
extractor = featureextractor.RadiomicsFeatureExtractor()
print(f"Extractor ready. Processing from: {image_folder}")

# === STEP 3: Loop through cases ===
all_features = []

for image_file in sorted(os.listdir(image_folder)):
    if not image_file.endswith(".nii.gz"):
        continue

    base_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
    mask_candidates = [m for m in os.listdir(mask_folder) if m.startswith(base_name)]
    if not mask_candidates:
        print(f" No mask found for {base_name}, skipping.")
        continue

    mask_path = os.path.join(mask_folder, mask_candidates[0])
    image_path = os.path.join(image_folder, image_file)

    # Load image & mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Visualization
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)
    z_indices = np.where(mask_np.sum(axis=(1, 2)) > 0)[0]
    if len(z_indices) == 0:
        print(f" Empty mask for {base_name}, skipping visualization.")
        continue
    mid_slice = z_indices[len(z_indices) // 2]

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np[mid_slice], cmap='gray')
    plt.contour(mask_np[mid_slice], colors='r', linewidths=1)
    plt.title(f"{base_name} (slice {mid_slice})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Extract features
    print(f" Extracting features for {base_name} ...")
    result = extractor.execute(image, mask)

    # Convert to dict and add metadata
    result_dict = {k: v for k, v in result.items()}
    result_dict["Case"] = base_name
    result_dict["label"] = 0  # default placeholder
    all_features.append(result_dict)

# === STEP 4: Save CSV ===
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved all features to: {output_csv}")
    print(f" Columns: {df.shape[1]}, Samples: {df.shape[0]}")
else:
    print("No valid cases processed.")

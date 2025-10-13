"""
PyRadiomics Batch Feature Extraction + Visualization
Author: doffi
Year: 2025
Description:
  This script extracts radiomics features from all CT images
  and their corresponding masks using relative paths, and
  visualizes each case interactively (one at a time).
"""

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor
# STEP 1: Resolve project paths 自动获取项目路径
# Get the directory where this script is located
current_dir = os.path.dirname(__file__)
# Go up two levels to reach the project root directory
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Define dataset root folder (dataset/sample/CT)
dataset_root = os.path.join(project_root, "dataset", "sample", "CT")
# Define image and mask subfolders
image_folder = os.path.join(dataset_root, "image")
mask_folder  = os.path.join(dataset_root, "nodule_mask")

# STEP 2: Initialize PyRadiomics feature extractor 初始化放射组学特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor()

# STEP 3: Process each case 逐个处理病例
for image_file in sorted(os.listdir(image_folder)):
    if not image_file.endswith(".nii.gz"):
        continue

    base_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
    mask_candidates = [m for m in os.listdir(mask_folder) if m.startswith(base_name)]
    if not mask_candidates:
        print(f" No mask found for {base_name}, skipping.")
        continue

    mask_file = os.path.join(mask_folder, mask_candidates[0])
    image_path = os.path.join(image_folder, image_file)

    # Load image & mask 读取影像与掩膜
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_file)

    # Convert to numpy arrays for visualization
    # 转为 numpy 数组用于显示
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)

    # Select middle slice of mask 选择掩膜中间切片
    z_indices = np.where(mask_np.sum(axis=(1,2)) > 0)[0]
    if len(z_indices) == 0:
        print(f" Empty mask for {base_name}, skipping visualization.")
        continue
    mid_slice = z_indices[len(z_indices)//2]


    # STEP 4: Visualization 图像可视化
    plt.figure(figsize=(6,6))
    plt.imshow(image_np[mid_slice], cmap='gray')
    plt.contour(mask_np[mid_slice], colors='r', linewidths=1)
    plt.title(f"{base_name} (slice {mid_slice})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()  # Wait for you to close before continuing 等你关掉图像再继续

    # STEP 5: Extract features 特征提取
    print(f" Extracting features for {base_name} ...")
    result = extractor.execute(image, mask)
    print(f" {base_name}: extracted {len(result)} features.\n")

print(" All cases processed successfully!")

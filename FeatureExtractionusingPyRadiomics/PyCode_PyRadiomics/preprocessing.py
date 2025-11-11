# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline for CT Images and Masks
Author: doffi

Description
-----------
This script standardizes CT images and masks for segmentation and classification:
  1) Resamples to a uniform voxel spacing (default: 1.0 x 1.0 x 1.0 mm)
  2) Performs intensity normalization (clip HU to [-1000, 400] and scale to [0, 1])
  3) Saves to a processed dataset folder that mirrors the input structure

Glossary:
  Resample      = standardize voxel spacing
  Normalize     = intensity normalization
  Spacing       = voxel spacing
  Origin/Direction = spatial reference info in the image header
"""

import os
import sys
import numpy as np
import SimpleITK as sitk


# ------------------------ Path Resolution ------------------------
CURRENT_DIR = os.path.dirname(__file__)                               # ...\FeatureExtractionusingPyRadiomics\PyCode_PyRadiomics
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # ...\UAB-Decision-Making

if len(sys.argv) > 1:
    DATASET_ROOT = os.path.abspath(sys.argv[1])
else:
    # Default: dataset/sample/CT ,  remember change here
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "sample", "CT")

# Output folder (derived from input folder name, e.g., CT or VOIs)
INFERRED_NAME = os.path.basename(DATASET_ROOT.rstrip("\\/"))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "dataset", "processed", INFERRED_NAME)


# ------------------------ Core Functions ------------------------
def resample_image(img: sitk.Image,
                   new_spacing=(1.0, 1.0, 1.0),
                   is_mask: bool = False) -> sitk.Image:
    """
    Resample an image or mask to a target spacing.
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new image size to preserve physical dimensions
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    return resampler.Execute(img)


def normalize_intensity(img: sitk.Image,
                        hu_range=(-1000, 400)) -> sitk.Image:
    """
    Clip image intensities to a given HU range and scale to [0, 1].
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, hu_range[0], hu_range[1])
    arr = (arr - hu_range[0]) / float(hu_range[1] - hu_range[0])
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def preprocess_ct_and_mask(image_path: str,
                           mask_path: str,
                           output_image_path: str,
                           output_mask_path: str,
                           new_spacing=(1.0, 1.0, 1.0),
                           hu_range=(-1000, 400)):
    """
    Preprocess one image/mask pair and save results.
    """
    # Load input image (float) and mask (uint8)
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    # Step 1: Resample
    image_r = resample_image(image, new_spacing=new_spacing, is_mask=False)
    mask_r = resample_image(mask, new_spacing=new_spacing, is_mask=True)

    # Step 2: Intensity normalization (after resampling)
    image_n = normalize_intensity(image_r, hu_range=hu_range)

    # Step 3: Copy spatial info to ensure alignment
    image_n.CopyInformation(mask_r)

    # Step 4: Save
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    sitk.WriteImage(image_n, output_image_path)
    sitk.WriteImage(mask_r, output_mask_path)

    print(f"Preprocessed: {os.path.basename(image_path)}")


def preprocess_dataset(input_root: str,
                       output_root: str,
                       new_spacing=(1.0, 1.0, 1.0),
                       hu_range=(-1000, 400)):
    """
    Batch preprocess a dataset folder containing image/ and nodule_mask/.
    """
    image_dir = os.path.join(input_root, "image")
    mask_dir = os.path.join(input_root, "nodule_mask")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    output_image_dir = os.path.join(output_root, "image")
    output_mask_dir = os.path.join(output_root, "nodule_mask")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Iterate through all .nii.gz images and find corresponding masks
    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".nii.gz"):
            continue

        image_path = os.path.join(image_dir, fname)
        base = os.path.splitext(os.path.splitext(fname)[0])[0]  # e.g. LIDC-IDRI-0001

        # Find matching mask (supports *_R_1.nii.gz, etc.)
        candidates = [m for m in os.listdir(mask_dir)
                      if m.startswith(base) and m.endswith(".nii.gz")]
        if not candidates:
            print(f"⚠️ Missing mask for {base}, skipping.")
            continue

        mask_name = sorted(candidates)[0]
        mask_path = os.path.join(mask_dir, mask_name)

        out_img = os.path.join(output_image_dir, fname)
        out_msk = os.path.join(output_mask_dir, mask_name)

        preprocess_ct_and_mask(
            image_path, mask_path,
            out_img, out_msk,
            new_spacing=new_spacing,
            hu_range=hu_range
        )


# ------------------------ Entry Point ------------------------
if __name__ == "__main__":
    in_root = DATASET_ROOT
    out_root = DEFAULT_OUTPUT

    print(f"Input dataset : {in_root}")
    print(f"Output folder : {out_root}\n")

    preprocess_dataset(in_root, out_root)

    print("Preprocessing finished.")
    print(f"Processed data saved under: {out_root}")

# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline for CT Images and Masks
Author: doffi

Description
-----------
This script standardizes CT images and masks for segmentation and classification:
  1) Resamples to a uniform voxel spacing (default: 1.0 x 1.0 x 1.0 mm)
  2) Performs intensity normalization (clip HU to [-1000, 400] and scale to [0, 1])
  3) (Optional) Smooths the CT image (Gaussian or median filter)
  4) (Optional) Applies morphological opening + closing to segmentation masks
  5) Saves to a processed dataset folder that mirrors the input structure

Glossary:
  Resample        = standardize voxel spacing
  Normalize       = intensity normalization
  Spacing         = voxel spacing
  Origin/Direction = spatial reference info in the image header
"""

import os
import sys
import numpy as np
import SimpleITK as sitk


# ------------------------ Defaults / Hyperparameters ------------------------

# Default target spacing for all volumes
DEFAULT_SPACING = (1.0, 1.0, 1.0)

# Default HU range for clipping before normalization
DEFAULT_HU_RANGE = (-1000, 400)

# Default smoothing configuration (for CT images)
DEFAULT_SMOOTH_METHOD = "gaussian"   # "gaussian", "median", or None
DEFAULT_GAUSSIAN_SIGMA = 1.0         # in voxel units (SimpleITK variance = sigma^2)
DEFAULT_MEDIAN_RADIUS = 1            # neighborhood radius for median filter

# Default morphology configuration (for segmentation masks only)
DEFAULT_MORPH_RADIUS = 2             # structuring element radius for opening/closing
DEFAULT_APPLY_SMOOTHING = True
DEFAULT_APPLY_MORPHOLOGY = True


# ------------------------ Path Resolution ------------------------

CURRENT_DIR = os.path.dirname(__file__)                                # ...\FeatureExtractionusingPyRadiomics\PyCode_PyRadiomics
PROJECT_ROOT = CURRENT_DIR  # ...\UAB-Decision-Making

# Allow passing dataset root from the command line (must contain image/ and nodule_mask/)
if len(sys.argv) > 1:
    DATASET_ROOT = os.path.abspath(sys.argv[1])
else:
    # Default: dataset/sample/CT  (remember to change this for the exam)
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "full", "VOIs")

# Output folder (derived from input folder name, e.g., CT or VOIs)
INFERRED_NAME = os.path.basename(DATASET_ROOT.rstrip("\\/"))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "dataset", "processed", INFERRED_NAME)


# ------------------------ Core Utility Functions ------------------------

def resample_image(
    img: sitk.Image,
    new_spacing=DEFAULT_SPACING,
    is_mask: bool = False
) -> sitk.Image:
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
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )

    return resampler.Execute(img)


def normalize_intensity(
    img: sitk.Image,
    hu_range=DEFAULT_HU_RANGE
) -> sitk.Image:
    """
    Clip image intensities to a given HU range and scale to [0, 1].
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, hu_range[0], hu_range[1])
    arr = (arr - hu_range[0]) / float(hu_range[1] - hu_range[0])

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def smooth_image(
    img: sitk.Image,
    method: str = DEFAULT_SMOOTH_METHOD,
    value: int | float = DEFAULT_GAUSSIAN_SIGMA,
) -> sitk.Image:
    """
    Apply smoothing to the CT image.

    method:
        - "gaussian": Discrete Gaussian filter (recommended)
        - "median"  : Median filter
        - None or unknown string: no smoothing
    """
    if method is None:
        return img

    method = method.lower()

    if method == "gaussian":
        # SimpleITK expects variance = sigma^2
        return sitk.DiscreteGaussian(img, variance=value**2)
    elif method == "median":
        return sitk.Median(img, [value] * 3)
    else:
        # Fallback: no smoothing if method is not recognized
        return img


def clean_mask_morphology(
    mask: sitk.Image,
    radius: int = DEFAULT_MORPH_RADIUS
) -> sitk.Image:
    """
    Apply morphological opening followed by closing to a binary mask.

    This removes small isolated regions and fills small holes, which is
    useful for segmentation masks.
    """
    radius_vec = [radius] * 3
    opened = sitk.BinaryMorphologicalOpening(mask, radius_vec)
    closed = sitk.BinaryMorphologicalClosing(opened, radius_vec)
    return closed


# ------------------------ High-level Preprocessing ------------------------

def preprocess_ct_and_mask(
    image_path: str,
    mask_path: str,
    output_image_path: str,
    output_mask_path: str,
    new_spacing=DEFAULT_SPACING,
    hu_range=DEFAULT_HU_RANGE,
    apply_smoothing: bool = DEFAULT_APPLY_SMOOTHING,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    smooth_value: int | float = DEFAULT_GAUSSIAN_SIGMA,
    morphology_radius: int = DEFAULT_MORPH_RADIUS,
    apply_morphology: bool = DEFAULT_APPLY_MORPHOLOGY,
    verbose: bool = False
):
    """
    Preprocess one image/mask pair and save results.

    Steps:
      1) Resample image and mask to the same spacing
      2) Intensity normalization (image only)
      3) Optional smoothing (image only)
      4) Optional morphological opening + closing (mask only)
      5) Save preprocessed image and mask
    """
    # Load input image (float) and mask (uint8 / binary)
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    # Step 1: Resample
    image_r = resample_image(image, new_spacing=new_spacing, is_mask=False)
    mask_r = resample_image(mask, new_spacing=new_spacing, is_mask=True)

    # Step 2: Intensity normalization (after resampling)
    image_n = normalize_intensity(image_r, hu_range=hu_range)

    # Step 3: Optional smoothing (for CT image only)
    if apply_smoothing and smooth_method is not None:
        image_n = smooth_image(
            image_n,
            method=smooth_method,
            value=smooth_value,
        )

    # Step 4: Optional morphological cleaning (for segmentation mask only)
    if apply_morphology and morphology_radius is not None and morphology_radius > 0:
        mask_c = clean_mask_morphology(mask_r, radius=morphology_radius)
    else:
        mask_c = mask_r

    # Step 5: Ensure spatial metadata stays aligned
    image_n.CopyInformation(mask_c)

    # Step 6: Save
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    sitk.WriteImage(image_n, output_image_path)
    sitk.WriteImage(mask_c, output_mask_path)

    if verbose:
        print(f"Preprocessed: {os.path.basename(image_path)}")


def preprocess_dataset(
    input_root: str,
    output_root: str,
    new_spacing=DEFAULT_SPACING,
    hu_range=DEFAULT_HU_RANGE,
    apply_smoothing: bool = DEFAULT_APPLY_SMOOTHING,
    smooth_method: str = DEFAULT_SMOOTH_METHOD,
    smooth_value: int | float = DEFAULT_GAUSSIAN_SIGMA,
    morphology_radius: int = DEFAULT_MORPH_RADIUS,
    apply_morphology: bool = DEFAULT_APPLY_MORPHOLOGY,
):
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
        candidates = [
            m for m in os.listdir(mask_dir)
            if m.startswith(base) and m.endswith(".nii.gz")
        ]
        if not candidates:
            print(f"Missing mask for {base}, skipping.")
            continue

        mask_name = sorted(candidates)[0]
        mask_path = os.path.join(mask_dir, mask_name)

        out_img = os.path.join(output_image_dir, fname)
        out_msk = os.path.join(output_mask_dir, mask_name)

        preprocess_ct_and_mask(
            image_path=image_path,
            mask_path=mask_path,
            output_image_path=out_img,
            output_mask_path=out_msk,
            new_spacing=new_spacing,
            hu_range=hu_range,
            apply_smoothing=apply_smoothing,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            morphology_radius=morphology_radius,
            apply_morphology=apply_morphology,
        )


# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    in_root = DATASET_ROOT
    out_root = DEFAULT_OUTPUT

    print(f"Input dataset : {in_root}")
    print(f"Output folder : {out_root}\n")

    preprocess_dataset(
        input_root=in_root,
        output_root=out_root,
        new_spacing=DEFAULT_SPACING,
        hu_range=DEFAULT_HU_RANGE,
        apply_smoothing=DEFAULT_APPLY_SMOOTHING,
        smooth_method=DEFAULT_SMOOTH_METHOD,
        smooth_value=DEFAULT_GAUSSIAN_SIGMA,
        morphology_radius=DEFAULT_MORPH_RADIUS,
        apply_morphology=DEFAULT_APPLY_MORPHOLOGY,
    )

    print("Preprocessing finished.")
    print(f"Processed data saved under: {out_root}")

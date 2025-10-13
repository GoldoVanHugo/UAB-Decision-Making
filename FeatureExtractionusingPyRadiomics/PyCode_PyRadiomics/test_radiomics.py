"""
PyRadiomics Feature Extraction Example
Author: doffi
Description:
  This script demonstrates how to extract radiomics features
  from a CT image and its corresponding mask using PyRadiomics.
"""

from radiomics import featureextractor
import SimpleITK as sitk
import os

# 1 edit your path
image_path = r"E:\Projects\UAB-Decision-Making\dataset\sample\CT\image\LIDC-IDRI-0001.nii.gz"
mask_path  = r"E:\Projects\UAB-Decision-Making\dataset\sample\CT\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz"

# 2 Check whether both files exist
print("Image exists:", os.path.exists(image_path))
print("Mask exists:", os.path.exists(mask_path))

# 3 Load the image and mask using SimpleITK
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

# 4 Initialize the PyRadiomics feature extractor (using default settings)
extractor = featureextractor.RadiomicsFeatureExtractor()

# 5 Perform feature extraction
result = extractor.execute(image, mask)

# 6 Display results
print(f"Successfully extracted {len(result)} features!")
print("First 10 feature names:")
print(list(result.keys())[:10])

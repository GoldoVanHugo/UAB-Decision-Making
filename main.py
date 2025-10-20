"""
Main entry point for Radiomics Pipeline
Author: doffi
Description:
  This script runs the full radiomics pipeline.
  1. Feature extraction using PyRadiomics.
  2. Classification using machine learning.
  You can specify the dataset path as a command-line argument.
  Example:
      python main.py dataset/full/VOIs
  If no argument is provided, it defaults to dataset/sample/CT.
"""

import os
import sys
import subprocess

# --- STEP 1: Determine dataset path (确定数据集路径) ---
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    dataset_path = os.path.join("dataset", "sample", "CT")  # default path

dataset_path = os.path.abspath(dataset_path)
print(f"\nUsing dataset: {dataset_path}")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

# --- STEP 2: Define sub-script paths (定义子脚本路径) ---
project_root = os.path.dirname(__file__)
extract_script = os.path.join(
    project_root, "FeatureExtractionusingPyRadiomics", "PyCode_PyRadiomics", "test_radiomics.py"
)
classify_script = os.path.join(project_root, "Classification", "Classification.py")

# --- STEP 3: Run feature extraction (调用特征提取脚本) ---
print("\nSTEP 1/2: Extracting radiomics features...")
subprocess.run(["python", extract_script, dataset_path], check=True)

# --- STEP 4: Run classification (调用分类脚本) ---
print("\nSTEP 2/2: Running classification model...")
subprocess.run(["python", classify_script, dataset_path], check=True)

print("\nPipeline completed successfully.")

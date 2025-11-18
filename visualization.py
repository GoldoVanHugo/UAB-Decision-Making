import os
import numpy as np

from io_data import readNifty

from joblib import load
from constants import SIZE_3D
from utils import (
    get_pat_and_node_id,
    patching,
    standard_scaling,
)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


if __name__ == "__main__":
    img_path = os.path.join("dataset", "processed", "VOIs", "image", "LIDC-IDRI-1011_R_3.nii.gz")
    mask_path = os.path.join("dataset", "processed", "VOIs", "nodule_mask", "LIDC-IDRI-1011_R_3.nii.gz")
    d3 = False

    img, _ = readNifty(img_path)
    mask, _ = readNifty(mask_path)

    model_path = os.path.join("models", "segmentation", "svm_test_251117.joblib")
    model = load(model_path)

    visu_folder = "visualization"
    pat_id, node_id = get_pat_and_node_id(file=img_path)

    for z in range(img.shape[-1]):
        org_img = img[..., z]
        org_mask = mask[..., z]
        if d3:
            img_ = patching(
                image=org_img,
                size_3d=SIZE_3D
            ).reshape((-1, SIZE_3D * SIZE_3D))
        else:
            img_ = org_img.reshape((-1, 1))

        y_pred = model.predict(img_)
        y_pred_mask = y_pred.reshape(org_img.shape)

        # 0 = TN (gray), 1 = FP (red), 2 = FN (blue), 3 = TP (green)
        error_map = np.zeros_like(org_img)
        error_map[(org_mask == 1) & (y_pred_mask == 1)] = 3  # TP
        error_map[(org_mask == 0) & (y_pred_mask == 0)] = 0  # TN
        error_map[(org_mask == 0) & (y_pred_mask == 1)] = 1  # FP
        error_map[(org_mask == 1) & (y_pred_mask == 0)] = 2  # FN

        cmap = ListedColormap([
            (0, 0, 0, 0),  # TN = transparent
            (1, 0, 0, 0.6),  # FP = rot
            (0, 0, 1, 0.6),  # FN = blau
            (0, 1, 0, 0.6),  # TP = grün
        ])

        store_path = os.path.join(visu_folder, pat_id, node_id)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Ground Truth Maske")
        plt.imshow(org_img, cmap="gray")
        plt.imshow(org_mask, cmap="Reds", alpha=0.5)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Prediction")
        plt.imshow(org_img, cmap="gray")
        plt.imshow(error_map, cmap=cmap)
        plt.axis("off")

        legend_elements = [
            Patch(facecolor=(0, 1, 0, 0.6), edgecolor='k', label='True Positive'),
            Patch(facecolor=(1, 0, 0, 0.6), edgecolor='k', label='False Positive'),
            Patch(facecolor=(0, 0, 1, 0.6), edgecolor='k', label='False Negative'),
            Patch(facecolor=(0, 0, 0, 0), edgecolor='k', label='True Negative')
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.1, -0.05), ncol=2, frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(store_path, f"image_{z}.png"))

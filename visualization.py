import os
from glob import glob
import numpy as np
import pandas as pd
from io_data import readNifty


from constants import SIZE_3D
from utils import (
    get_pat_and_node_id,
    patching,
)
from trainer import (
    TrainerSVM,
    TrainerRandomForest
)
from dataloader.dataloader_base import DataloaderBase
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

META = os.path.join("dataset", "full", "MetadatabyNoduleMaxVoting.xlsx")


def get_best_values(csv_file: str, sort_by: str = "ROC-AUC"):
    df = pd.read_csv(csv_file)
    df_sortiert = df.sort_values(by=sort_by, ascending=False)

    print(df_sortiert.iloc[0])
    print(df_sortiert.iloc[1])
    print(df_sortiert.iloc[3])

    return df_sortiert.iloc[0]


def get_trainer_with_model(model_type: str, experiment_folder: str):
    if model_type == "svm":
        train = TrainerSVM("", "")
    else:
        train = TrainerRandomForest("", "")

    model_path = glob(os.path.join(experiment_folder, "model", f"*{train.model_extension()}"))[0]
    train.load_model(model_path=model_path)

    return train


def get_test_paths(experiment_folder: str):
    data_path = os.path.join(os.path.dirname(experiment_folder), "data")
    dataload = DataloaderBase(data_path=data_path, meta_file_path=META)
    _, test_paths = dataload.get_train_and_test_paths()

    return test_paths


def seg_visualization(img_path: str, mask_path: str, trainer, d3, size_3d: int, output_folder: str):
    img, _ = readNifty(img_path)
    mask, _ = readNifty(mask_path)

    pat_id, nodule_id = get_pat_and_node_id(file=img_path)

    for z in range(img.shape[-1]):
        org_img = img[..., z]
        org_mask = mask[..., z]
        if d3:
            img_ = patching(
                image=org_img,
                size_3d=size_3d
            ).reshape((-1, size_3d * size_3d))
        else:
            img_ = org_img.reshape((-1, 1))

        _, y_pred = trainer.predict(img_)
        y_pred_mask = y_pred.reshape(org_img.shape)

        create_plot(
            store_path=os.path.join(output_folder, pat_id, nodule_id),
            org_img=org_img,
            org_mask=org_mask,
            y_pred_mask=y_pred_mask,
            z=z
        )


def class_visualization(img_path: str, mask_path: str, trainer, d3, size_3d: int, output_folder: str):
    img, _ = readNifty(img_path)
    mask, _ = readNifty(mask_path)
    meta_data = pd.read_excel(META)

    pat_id, nodule_id = get_pat_and_node_id(file=img_path)
    pat_rows = meta_data.loc[meta_data["patient_id"] == pat_id]
    pat_meta_data = pat_rows.loc[pat_rows["nodule_id"] == int(nodule_id.split("_")[-1])]

    for z in range(img.shape[-1]):
        org_img = img[..., z]
        org_mask = mask[..., z]
        org_mask_flatten = org_mask.flatten()
        count = np.count_nonzero(org_mask_flatten)
        if count == 0:
            continue

        if d3:
            img_ = patching(
                image=org_img,
                size_3d=size_3d
            ).reshape((-1, size_3d * size_3d))
        else:
            img_ = org_img.reshape((-1, 1))

        x = img_[org_mask_flatten == 1]
        y_true_flat = np.full(shape=(count,), fill_value=pat_meta_data["Diagnosis_value"], dtype=int)

        _, y_pred_flat = trainer.predict(x)
        y_pred = np.full(shape=org_mask.shape, fill_value=-1)
        y_pred[org_mask == 1] = y_pred_flat
        y_true = np.full(shape=org_mask.shape, fill_value=-1)
        y_true[org_mask == 1] = y_true_flat

        create_plot(
            store_path=os.path.join(output_folder, pat_id, nodule_id),
            org_img=org_img,
            org_mask=y_true,
            y_pred_mask=y_pred,
            z=z
        )


def create_plot(store_path: str, org_img, org_mask, y_pred_mask, z):
    # 0 = TN (gray), 1 = FP (red), 2 = FN (blue), 3 = TP (green)
    error_map = np.full(shape=org_img.shape, fill_value=-1)
    error_map[(org_mask == 1) & (y_pred_mask == 1)] = 3  # TP
    error_map[(org_mask == 0) & (y_pred_mask == 0)] = 0  # TN
    error_map[(org_mask == 0) & (y_pred_mask == 1)] = 1  # FP
    error_map[(org_mask == 1) & (y_pred_mask == 0)] = 2  # FN

    cmp_list = [
        (0, 0, 0, 0),  # TN = transparent
        (1, 0, 0, 0.6),  # FP = rot
        (0, 0, 1, 0.6),  # FN = blau
        (0, 1, 0, 0.6),  # TP = grün
    ]
    legend_elements = [
        Patch(facecolor=(0, 1, 0, 0.6), edgecolor='k', label='True Positive'),
        Patch(facecolor=(1, 0, 0, 0.6), edgecolor='k', label='False Positive'),
        Patch(facecolor=(0, 0, 1, 0.6), edgecolor='k', label='False Negative'),
        Patch(facecolor=(0, 0, 0, 0), edgecolor='k', label='True Negative')
    ]

    if np.count_nonzero(error_map == -1) != 0:
        cmp_list.insert(0,(1, 1, 0, 0.4)) # not used
        legend_elements.append(
            Patch(facecolor=(1, 1, 0, 0.4), edgecolor='k', label='Not predicted')
        )

    cmap = ListedColormap(cmp_list)

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

    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.1, -0.05), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(store_path, f"image_{z}.png"))
    plt.close()


if __name__ == "__main__":
    best_seg = get_best_values(csv_file=os.path.join("experiments", "segmentation", "results.csv"))
    seg_test_X, seg_test_y = get_test_paths(best_seg["Experiment_folder"])
    seg_trainer = get_trainer_with_model(model_type=best_seg["model_type"], experiment_folder=best_seg["Experiment_folder"])

    for X, y in zip(seg_test_X, seg_test_y):
        seg_visualization(
            img_path=X,
            mask_path=y,
            d3=best_seg["3D"],
            trainer=seg_trainer,
            size_3d=best_seg["3D size"],
            output_folder=os.path.join(best_seg["Experiment_folder"], "visualization")
        )

    best_class = get_best_values(csv_file=os.path.join("experiments", "classification", "results.csv"))
    class_test_X, class_test_y = get_test_paths(best_class["Experiment_folder"])
    class_trainer = get_trainer_with_model(model_type=best_class["model_type"],
                                         experiment_folder=best_class["Experiment_folder"])

    for X, y in zip(class_test_X, class_test_y):
        class_visualization(
            img_path=X,
            mask_path=y,
            d3=best_class["3D"],
            trainer=class_trainer,
            size_3d=best_class["3D size"],
            output_folder=os.path.join(best_class["Experiment_folder"], "visualization")
        )
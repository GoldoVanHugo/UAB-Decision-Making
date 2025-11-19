import itertools
import os
import shutil
import csv

from dataloader import DataloaderSegmentation
from trainer import (
    TrainerRandomForest,
    TrainerSVM,
)
from preprocessing import preprocess_dataset


PROJECT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "full")
RAW_DATA_PATH = os.path.join(DATASET_PATH, "VOIs")
META_FILE_PATH = os.path.join(DATASET_PATH, "MetadatabyNoduleMaxVoting.xlsx")

BASE_PREPROCESS_CONFIG = {
    "new_spacing": (1.0, 1.0, 1.0),
    "hu_range": (-1000, 400),
    "apply_smoothing": True,
    "smooth_method": "gaussian", # "gaussian", "median", or None
    "smooth_value": 1.0,
    "apply_morphology": False,
    "morphology_radius": 0
}

BASE_DATALOADER_CONFIG = {
    "meta_file_path": META_FILE_PATH,
    "set_seed": True,
    "use_all_voxels": True,
    "d3": False,
    "size_3d": 0,
}
BASE_TRAINER_CONFIG = {
    "model_name": "model",
    "set_seed": True,
}


def get_experiment_preprocess_configs():
    configs = []

    smoothing_gaussian_values = [2.0, 5.0]
    smoothing_median_values = [2, 5]

    smoothing_options = list(itertools.product([True], ["gaussian"], smoothing_gaussian_values))
    smoothing_options += list(itertools.product([True], ["median"], smoothing_median_values))
    # add no smoothing to options
    smoothing_options.insert(0, (False, None, None))

    for (s_apply, s_method, s_val) in smoothing_options:
        config = BASE_PREPROCESS_CONFIG.copy()
        config["apply_smoothing"] = s_apply
        config["smooth_method"] = s_method
        config["smooth_value"] = s_val

        configs.append(config)

    return configs

def get_experiment_training_configs():
    model_types = ["svm", "rf"]
    sizes_3d = [3, 5]
    configs = [
        {
            "model_type": "svm",
            "dataloader": BASE_DATALOADER_CONFIG.copy(),
            "trainer": BASE_TRAINER_CONFIG.copy()
        },
        {
            "model_type": "rf",
            "dataloader": BASE_DATALOADER_CONFIG.copy(),
            "trainer": BASE_TRAINER_CONFIG.copy()
        }
    ]

    for model_type, size_3d in itertools.product(model_types, sizes_3d):
        dataloader_config = BASE_DATALOADER_CONFIG.copy()
        dataloader_config["d3"] = True
        dataloader_config["size_3d"] = size_3d
        configs.append({
            "model_type": model_type,
            "dataloader": dataloader_config,
            "trainer": BASE_TRAINER_CONFIG.copy()
        })

    return configs


def experiment():
    experiment_name = "classification"
    experiment_root = os.path.join(PROJECT_DIR, "experiments", experiment_name)

    if not os.path.exists(experiment_root):
        os.makedirs(experiment_root)

    prepro_configs = get_experiment_preprocess_configs()

    results = []

    for idx, prepro_config in enumerate(prepro_configs):
        train_configs = get_experiment_training_configs()
        idxs = (idx * len(train_configs) + 1, idx * len(train_configs) + len(train_configs))
        print(f"------- Start preprocessing for experiment {idxs[0]} - {idxs[1]} --------")
        print(f"With config: {prepro_config}")
        experiments_folder = os.path.join(experiment_root, f"experiments_{idxs[0]}-{idxs[1]}")
        if os.path.exists(experiments_folder):
            shutil.rmtree(experiments_folder)

        os.makedirs(experiments_folder)

        prepro_data_path = os.path.join(experiments_folder, "data")

        preprocess_dataset(
            input_root=RAW_DATA_PATH,
            output_root=prepro_data_path,
            **prepro_config
        )

        for t_idx, train_config in enumerate(train_configs):
            print(f"----- Start experiment {idxs[0] + t_idx} -----")
            print(f"With configs: {train_config}")
            experiment_folder = os.path.join(experiments_folder, f"experiment_{idxs[0] + t_idx}")
            train_config["trainer"]["model_path"] = os.path.join(experiment_folder, "model")

            dataloader = DataloaderSegmentation(
                data_path=prepro_data_path,
                **train_config["dataloader"]
            )
            if train_config["model_type"] == "rf":
                trainer_class = TrainerRandomForest
            else:
                trainer_class = TrainerSVM

            trainer = trainer_class(**train_config["trainer"])

            train_paths, test_paths = dataloader.get_train_and_test_paths()
            train_dataset, test_dataset = dataloader.get_train_and_test_datasets(
                train_paths=train_paths,
                test_paths=test_paths
            )
            print("--- Start Training ---")
            trainer.train(
                x=train_dataset[0],
                y=train_dataset[1],
            )
            print("--- Start Test ---")
            metrics = trainer.test(
                x_test=test_dataset[0],
                y_test=test_dataset[1],
            )
            trainer.save_model()

            results.append({
                "Experiment_folder": experiment_folder,
                "new_spacing": prepro_config["new_spacing"],
                "hu_range": prepro_config["hu_range"],
                "apply_smoothing": prepro_config["apply_smoothing"],
                "smooth_method": prepro_config["smooth_method"],
                "smooth_value": prepro_config["smooth_value"],
                "model_type": train_config["model_type"],
                "3D": train_config["dataloader"]["d3"],
                "3D size": train_config["dataloader"]["size_3d"],
                "accuracy": metrics[0],
                "precision": metrics[1],
                "recall": metrics[2],
                "F1-Score": metrics[3],
                "ROC-AUC": metrics[4]
            })

    with open(os.path.join(experiment_root, "results.csv"), mode="w", newline="", encoding="utf-8") as file:
        fieldnames = results[0].keys()

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerows(results)


if __name__ == "__main__":
    experiment()

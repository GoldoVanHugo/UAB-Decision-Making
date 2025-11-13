import os

from dataloader import DataloaderSegmentation
from trainer import (
    TrainerRandomForest,
    TrainerSVM,
)

DIR_PATH = os.path.dirname(__file__)

config_benny = {
    "model_type": "rf",  # rf = RandomForest, svm = SVM
    "dataloader": {
        "data_path": os.path.join(DIR_PATH, "dataset", "full", "VOIs"),
        "meta_file": "MetadatabyNoduleMaxVoting.xlsx",
        "set_seed": True,
        "use_all_voxels": False,
        "d3": True,    # set to true for RandomForest
    },
    "trainer": {
        "model_path": os.path.join(DIR_PATH, "models", "segmentation"),
        "model_name": "rf_test_251111",
        "set_seed": True,
    },
    "load_model": False,
    # "load_model_path": "...",
    "pipline_steps": {
        "train": True,
        "test": True,
        "save_model": True,
    },
}

if __name__ == "__main__":
    config = config_benny
    dataloader = DataloaderSegmentation(**config["dataloader"])
    if config["model_type"] == "rf":
        trainer_class = TrainerRandomForest
    else:
        trainer_class = TrainerSVM

    trainer = trainer_class(**config["trainer"])

    if config["load_model"]:
        if "load_model_path" not in config:
            raise ValueError("Set the key 'load_model_path' to load a model.")
        trainer.load_model(model_path=config["load_model_path"])

    train_dataset, test_dataset = dataloader.get_train_and_test_datasets()

    if config["pipline_steps"]["train"]:
        print("----- Start Training -----")
        trainer.train(
            x=train_dataset[0],
            y=train_dataset[1],
        )
        print("----- Finish Training -----")

    if config["pipline_steps"]["test"]:
        print("----- Start Test -----")
        trainer.test(
            x_test=test_dataset[0],
            y_test=test_dataset[1],
        )
        print("----- Finish Test -----")

    if config["pipline_steps"]["save_model"]:
        trainer.save_model()

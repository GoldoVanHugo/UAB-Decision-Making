import os

from trainer.classification.trainer_classification import TrainerClassification

REPO_PATH = os.path.dirname(__file__)


config_benny = {
    "model_type": "svm",  # rf = RandomForest, svm = SVM
    "trainer": {
        "data_path": os.path.join(REPO_PATH, "dataset", "full", "VOIs"),
        "model_path": os.path.join(REPO_PATH, "models", "classification"),
        "model_name": "svm_test_251015",
        "meta_file": "MetadatabyNoduleMaxVoting.xlsx"
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
    """if config["model_type"] == "rf":
        trainer_class = TrainerRandomForest
    else:
        trainer_class = TrainerSVM"""

    trainer_class = TrainerClassification

    trainer = trainer_class(**config["trainer"])

    if config["load_model"]:
        if "load_model_path" not in config:
            raise ValueError("Set the key 'load_model_path' to load a model.")
        trainer.load_model(model_path=config["load_model_path"])

    train_dataset, test_dataset = trainer.get_train_and_test_datasets()

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

import os.path

from trainer import (
    TrainerRandomForest,
    TrainerSVM,
)

DIR_PATH = os.path.dirname(__file__)

config_benny = {
    "data_path": os.path.join(DIR_PATH, "dataset", "sample", "CT"),
    "model_path": os.path.join(DIR_PATH, "models"),
    "model_name": "svm_test_251015",
    "load_model": False,
    "model_type": "svm",  # rf = RandomForest, svm = SVM
    # "load_model_path": "...",
    "pipline_steps": {
        "train": True,
        "test": True,
        "save_model": True,
    },
}

if __name__ == "__main__":
    config = config_benny
    if config["model_type"] == "rf":
        trainer_class = TrainerRandomForest
    else:
        trainer_class = TrainerSVM

    trainer = trainer_class(
        data_path=config["data_path"],
        model_path=config["model_path"],
        model_name=config["model_name"],
    )

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

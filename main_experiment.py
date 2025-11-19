import classification_experiment
import segmentation_experiment


if __name__ == "__main__":
    print("------ Start Segmentation experiment ------")
    segmentation_experiment.experiment()
    print("------ Start Classification experiment ------")
    classification_experiment.experiment()

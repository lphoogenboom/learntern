# Import necessary libraries
from learner import Learner
from dataProcessor import dataProcessor
import numpy as np

if __name__ == "__main__":
    print('=== RAN AS FILE [learnter/main.py] ===')

    # Instantiate learner
    learner = Learner()

    # Get arguments from the command line, such as config file, etc.

    # Define device (cpu, cuda, mps, ...)
    Learner.setDevice("cpu")

    # Define input/output paths
    here = dataProcessor().getParent()
    path_splits = here/"data"/"splits"/"5-fold-indices.npz"
    path_images = here/"data"/"arrays"/"images.npy"
    path_labels = here/"data"/"arrays"/"labels.npy"
    path_config = here/"configuration.json"

    path_plots = here/"output"/"plots"
    path_weights = here/"outputs"/"weights"
    path_logs = here/"outputs"/"logs"


    # Define or load configuration
    learner.setConfig(path_config)

    # Create log and add the current configuration to it
    log = dict(
            name=run,
            epoch="",
            accuracy_train=[],
            accuracy_val=[],
            loss_train=[],
            loss_val=[],
            time_train=[],
            time_val=[],
            saved_weights=[],
            configuration=learner.configuration,
            device=learner.device,
        )
    
    # Load datasets
    images = np.load(path_images)
    labels = np.load(path_labels)
    splits = np.load(path_splits)

    dataset_train = Dataset(
            images=images[splits[f"train_{cv}"]],
            labels=labels[splits[f"train_{cv}"]],
        )

    # Load dataloaders

    # Load model

    # Load criterion

    # Load optimizer

    # Start training loop
    epochs = 100
    for epoch in range(100):
        # Train model
        step("train")

        # Validate model
        step("val")

        # Save model if best validation loss

        # Save logs

        # Save plots

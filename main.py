import os

import constants
import torch
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    dimensions = [200, 150]

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset()
    val_dataset = StartingDataset()
    model = StartingNetwork(dimensions)
    model = model.to(device)
    
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dimensions=dimensions,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()

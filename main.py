import os

import constants
import torch
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH = "/kaggle/input"

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    dimensions = [constants.BATCH_SIZE, 3, 200, 150]

    # Initalize dataset and model. Then train the model!
    
    # train_dataset = StartingDataset(0, (21000 * 0.8 // 32) * 32, PATH)
    # val_dataset = StartingDataset((21000 * 0.8 // 32) * 32, 21000 // 32 * 32, PATH)
    train_dataset = StartingDataset(0, 1920, PATH, device)
    val_dataset = StartingDataset(1920, 3840, PATH, device)
    model = StartingNetwork()
    model = model.to(device)
    
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dimensions=dimensions,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device
    )
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PATH = "data/"

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    dimensions = [constants.BATCH_SIZE, 3, 200, 150]

    # Initalize dataset and model. Then train the model!
    
    # train_dataset = StartingDataset(0, (21000 * 0.8 // 32) * 32, PATH)
    # val_dataset = StartingDataset((21000 * 0.8 // 32) * 32, 21000 // 32 * 32, PATH)
    train_dataset = StartingDataset(0, 1920, PATH)
    val_dataset = StartingDataset(1920, 3840, PATH)
    model = StartingNetwork()
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

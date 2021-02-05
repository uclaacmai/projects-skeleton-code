import argparse
import os
import time


import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import numpy as np
import torch

SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    count = 21397
    # count = 64
    train_prop = 0.70
    path = './cassava-leaf-disease-classification/train.csv'
    data = np.genfromtxt(path, delimiter=',', dtype='str')
    train_dataset = StartingDataset(truth = data[1:int(count*train_prop), 1], images = data[1:int(count*train_prop), 0], base = './cassava-leaf-disease-classification/train_images')
    val_dataset = StartingDataset(truth = data[int(count*train_prop):count, 1], images = data[int(count*train_prop):count, 0], base = './cassava-leaf-disease-classification/train_images')
    print(train_dataset[0])
    model = StartingNetwork(3, 5)
    model = model.to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument(
        "--n_eval", type=int, default=constants.N_EVAL,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

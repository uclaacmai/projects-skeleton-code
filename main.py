import argparse
import os
import time
import torch

import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import ConvNet
from train_functions.starting_train import starting_train


SUMMARIES_PATH = "training_summaries"


def main():
    images_dir = "./cassava-leaf-disease-classification/train_images"

    # Get command line arguments    
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(torch.cuda.get_device_name(0))

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!

    train_dataset = StartingDataset(images_dir, 'train.csv')
    val_dataset = StartingDataset(images_dir, 'val.csv')
    model = ConvNet(3, 5)
    #train_dataset.__showitem__(0)

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

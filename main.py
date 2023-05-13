import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import torchvision
import torch

import matplotlib.pyplot as plt

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}
    if torch.cuda.is_available():   #Select gpu if available
        device = torch.device('cuda')
        #"cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    

    #Need to define train_dataset and test_dataset

    

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset('../data/train.csv', transform=torchvision.transforms.ToTensor())

    img, lab = train_dataset.__getitem__(0)
    print(img)
    print(type(img))

    val_dataset = StartingDataset('../data/train.csv', transform=torchvision.transforms.ToTensor())
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )

    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)


    # Take a look at the first training example
    image, label = train_dataset[0] #POSE THIS AS A QUESTION - ASK IF PPL REMEMBER HOW TO GRAB THE FIRST TRAINING EXAMPLE
    #plt.imshow(image.squeeze(), cmap='gray') # Display grayscale image. 
    #print('Label:', label)
    print(image.shape)
    print(type(image))

if __name__ == "__main__":
    main()

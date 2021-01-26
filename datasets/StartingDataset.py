import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        #train_labels = pd.read_csv("/Users/advaithg/Documents/ACMAI-Projects-W21/cassava-leaf-disease-classification/train.csv")
        self.train_labels = pd.read_csv(path)

    def __getitem__(self, index):
        imgname = self.train_labels[index,0]
        inputs = "/Users/advaithg/Documents/ACMAI-Projects-W21/cassava-leaf-disease-classification/train_images/" + imgname
        img = Image.open(inputs)
        label = self.train_labels[index,1]
        return img, label

    def __showitem__(self, index):
        img, label = self.__getitem__(index)
        plt.imshow(img[0].squeeze(), cmap = 'gray')
        print(label)

    def __len__(self):
        return 10000


import torch
import os
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import torchvision

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, images_dir):
        self.csv_file = pd.read_csv('train.csv')
        self.images_dir = images_dir

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):

        image_id = os.path.join(self.images_dir,
                                self.csv_file.iloc[index, 0])
        image = io.imread(image_id)
        image = torch.Tensor(image)
        labels = self.csv_file.iloc[index, 1:]
        labels = torch.Tensor(labels)

        return image, labels

    def __showitem__(self, index):
        img, label = self.getitem(index)
        plt.imshow(img[0].squeeze(), cmap = 'gray')
        print(label)

import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        self.path = path
        self.images, self.labels = pd.read_csv(constants.DATA + "/train.csv")

    def __getitem__(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index])
        label = self.labels[index]

        image = image.resize(224, 448)
        image = ImageOps.grayscale(image)

        return image, label


    def __len__(self):
        return len(self.labels)

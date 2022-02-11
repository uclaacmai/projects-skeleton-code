import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants
import torchvision
import torchvision.transforms.functional
import numpy as np


class ImageAugment(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(constants.DATA + "/train.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.images = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1]
        self.transition = list(set(self.labels))
        self.whales = self.labels.replace(self.transition, list(range(5005)))

        self.transform1 = torchvision.transforms.RandomResizedCrop(size = (448,224), scale = (0.5, 0.75))

        self.transform2 = torchvision.transforms.ColorJitter()

        self.transform3 = torchvision.transforms.RandomAffine(180)

        self.augmentedimages = []
        self.augmentedlabels = []
            

    def __getitem__(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index])

        label = self.whales[index]

        image = image.resize((448, 224))
        image = ImageOps.grayscale(image)

        image = torchvision.transforms.ToTensor()(np.array(image))

        return image, label


    def __len__(self):
        return len(self.labels)

    def cutout(self, image):
        size = 30
        x = np.random.randint(448)
        y = np.random.randint(224)
        y1 = np.clip(y - size // 2, 0, 224)
        y2 = np.clip(y + size // 2, 0, 224)
        x1 = np.clip(x - size // 2, 0, 448)
        x2 = np.clip(x + size // 2, 0, 448)
        image[y1:y2, x1:x2] = 0



    def augment(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index])

        image = image.resize((448,224))

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.transform1(image))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.transform2(image))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.transform2(image))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.transform3(image))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.transform3(image))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(image + torch.std(image)*torch.randn(image.size))))
        self.augmentedlabels.append(index)

        self.augmentedimages.append(torchvision.transforms.ToTensor()(np.array(self.cutout(self, image))))




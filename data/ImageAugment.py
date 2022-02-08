import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants
import torchvision


class ImageAugment(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(constants.DATA + "/train.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.images = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1]
        self.transition = list(set(self.labels))
        self.whales = self.labels.replace(self.transition, list(range(5005)))

        self.transform1 = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(200),
            torchvision.transforms.function.resize((448,224))
        ])

        self.transform2 = torchvision.transforms.ColorJitter()

        self.transform3 = torchvision.transforms.RandomRotation(180)
            

    def __getitem__(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index])

        label = self.whales[index]

        image = image.resize((448, 224))
        image = ImageOps.grayscale(image)

        return torchvision.transforms.functional.pil_to_tensor(image), label


    def __len__(self):
        return len(self.labels)

    def augment(self, index);
        image, label = ImageAugment.__getitem__(index)

        self.images.append(self.transform1(image))
        self.labels.append(index)
        
        self.images.append(self.transform2(image))
        self.labels.append(index)

        self.images.append(self.transform2(image))
        self.labels.append(index)

        self.images.append(self.transform3(image))
        self.labels.append(index)

        self.images.append(self.transform3(image))
        self.labels.append(index)


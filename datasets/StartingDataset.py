import torch
from torchvision import transforms
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, images, truth, base):
        self.images = images
        self.truth = truth
        self.base = base

    def __getitem__(self, index):
        path = self.images[index]
        label = self.truth[index]

        img = Image.open(f'{self.base}/{path}')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = preprocess(img)
        return tensor, int(label)

    def __len__(self):
        return len(self.truth)

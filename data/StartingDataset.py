import torch
import PIL.Image as img
import numpy as np
import pandas as pd
import torchvision

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __getitem__(self, index):
        image = self.data.iloc[index, 0]
        image_real = img.open('../data/train_images/' + image)
        label = self.data.iloc[index, 1]
        
        #if self.transform is not None:
            #image = self.transform(image)
            
        #return image, label
    

        return (torchvision.transforms.ToTensor()(image_real), label)

    def __len__(self):
        return len(self.data)

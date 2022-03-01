from ctypes import resize
import constants
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import re
import random

INPUT_WIDTH=224
INPUT_HEIGHT=224

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, training_set, path, sample_factor):
        self.path = path
        df = pd.read_csv(path + "/cassava-leaf-disease-classification/train.csv")
        df = df.sort_values('label')
        print(df)
        self.training_set = training_set
        if (self.training_set):
            index_range = np.r_[0:800, 1087:2687, 3276:4876, 5662:15262, 18820:20420]
        else:
            index_range = np.r_[800:1087, 2686:3276, 4876:5662, 15262:18820, 20420:21375]

        self.pictures = df["image_id"].iloc[index_range].tolist()
        self.labels = df["label"].iloc[index_range].tolist()
        #convert to list

        if (self.training_set):
            x = len(self.pictures)
            for i in range(x):
                if(self.labels[i] != 3):
                    temp = self.pictures[i][:-4]
                    first_five_transformations = ["_r.jpg", "_gb.jpg", "_p.jpg", "_i.jpg", "_hf.jpg"]
                    for trans in first_five_transformations:
                        self.pictures.append(temp + trans)
                        self.labels.append(self.labels[i])
                    
                    if(self.labels[i] == 0):
                        last_six_transformations = ["_vf.jpg", "_s.jpg", "_so.jpg", "_r.jpg", "_cj.jpg","_r.jpg"]
                        for trans in last_six_transformations:
                            self.pictures.append(temp + trans)
                            self.labels.append(self.labels[i])

        size = int(len(self.pictures) * sample_factor)
        size = constants.BATCH_SIZE * math.ceil(size / constants.BATCH_SIZE)
        self.pictures, self.labels = zip(*random.sample(
            list(zip(self.pictures, self.labels)), size
        ))
                
        print(len(self.pictures))

        # if label is 0, 1, 2, 4: add transformations needed to balance data
        # append _something to the end of filename to specify transformation later
        # call .sample to shuffle data afterwards
            

    def __getitem__(self, index):
        # Grab a single training example
        picture = self.pictures[index]
        picture = re.sub(r'_[a-z]+',"",picture)
        # Load and resize the desired image
        im = self.resizeImage(self.path + "/cassava-leaf-disease-classification/train_images/" + picture)
        label = self.labels[index]
        trans = transforms.ToTensor()
        im = trans(im)
        if(self.pictures[index].endswith("_r.jpg")):
            rotater = transforms.RandomRotation(degrees=(0,180))
            im = rotater.forward(im)
        elif(self.pictures[index].endswith("_gb.jpg")):
            gb = transforms.GaussianBlur(3)
            im = gb.forward(im)
        elif(self.pictures[index].endswith("_p.jpg")):
            rp = transforms.RandomPerspective(p=1.0)
            im = rp.forward(im)
        elif(self.pictures[index].endswith("_i.jpg")):
            i = transforms.RandomInvert(1)
            im = i.forward(im)
        elif(self.pictures[index].endswith("_hf.jpg")):
            hf = transforms.RandomHorizontalFlip(p=1.0)
            im = hf.forward(im)
        elif(self.pictures[index].endswith("_vf.jpg")):
            vf = transforms.RandomVerticalFlip(p=1.0)
            im = vf.forward(im)
        elif(self.pictures[index].endswith("_s.jpg")):
            s = transforms.RandomAdjustSharpness(sharpness_factor=2,p=1.0)
            im = s.forward(im)
        elif(self.pictures[index].endswith("_so.jpg")):
            so = transforms.RandomSolarize(192.0,1.0)
            im = so.forward(im)
        elif(self.pictures[index].endswith("_cj.jpg")):
            cj = transforms.ColorJitter(brightness=.5,contrast=.5)
            im = cj.forward(im)
        
        # perform transformation if filename has _ in it
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize(im)
        example = (im, label)

        return example

    def __len__(self):
        return len(self.pictures)

    def resizeImage(self, im_path):
        im = Image.open(im_path)
        im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
        # im.show()
        return im

# if __name__ == "__main__":
#     datasetInstance = StartingDataset(0, 10, '.', 'cpu')
#     print(datasetInstance[0])
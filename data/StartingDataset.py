from ctypes import resize
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import re

INPUT_WIDTH=224
INPUT_HEIGHT=224

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, i, j, path, device):
        self.path = path
        df = pd.read_csv(path + "/cassava-leaf-disease-classification/train.csv")
        self.pictures = df["image_id"][i:j].tolist()
        self.labels = df["label"][i:j].tolist()
        self.device = device
        #convert to list

        x = len(self.pictures)
        for i in range(x):
            if(self.labels[i] != 3):
                temp = self.pictures[i][:-4]
                self.pictures.append(temp + "_r.jpg")
                self.labels.append(self.labels[i])
                
                self.pictures.append(temp + "_gb.jpg")
                self.labels.append(self.labels[i])

        # possible transformations: flip, rotate, translation, blur, brightness/hue (dunno if we want to mess with that), 
                
        print(len(self.pictures))

        # if label is 1, 2, 4, 0: add transformations needed to balance data
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
            im = transforms.functional.rotate(im, 45)
        # perform transformation if filename has _ in it
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize(im) # convert image to tensor first!
        if self.device != "cpu":
            im = im.cuda()
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
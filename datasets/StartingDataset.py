import torch
from torchvision.transforms import ToTensor, Lambda, transforms
import pandas as pd
import PIL
from PIL import Image as PImage
from os import listdir

class StatementDataset(torch.utils.data.Dataset):#inherit from torch.utils.data.Dataset to make our life easier in dealing with Data
    def __init__(self, statements, labels): #image_id, labels 
        self.statements = statements
        self.labels = labels 
    def __len__(self):
        return len(self.statements)
    def __getitem__(self, index):  #retrieve items from our dataset 
        path ='/Users/howard/Desktop/ACM-Project-Datasets/train_images/'
        trans1 = transforms.ToTensor()        
        statement = PImage.open(path+self.statements[index]) #read specific image
        statement = trans1(statement)
        label = self.labels[index]
        return (statement,label) #return a tuple 

image_info = pd.read_csv((r'/Users/howard/Desktop/ACM-Project-Datasets/train.csv'))
image_dataset = StatementDataset(image_info.image_id,image_info.label)

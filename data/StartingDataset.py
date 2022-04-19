import torch
import numpy
import constants
import pandas
from matplotlib import image
from matplotlib import pyplot
class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    ###create this function
    def __init__(self):
        self.csv_data = pandas.read_csv(constants.PATH_TO_DATA+'train.csv').to_numpy()
    def __getitem__(self, index):
        ## do loading here
        image_name, label = self.csv_data[index]
        inputs = image.imread('kolala.jpeg')
        print(inputs)
        
        return inputs, label

    def __len__(self):
        return 10000

import torch
import torchvision
import numpy
import constants
from PIL import Image
import pandas
from resizeimage import resizeimage
import matplotlib.pyplot as plt
# from matplotlib import image
# from matplotlib import pyplot
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

        # save image
        with Image.open(constants.PATH_TO_DATA+'/train_images/'+image_name) as inputs:
            inputs = torchvision.transforms.functional.resize(inputs, (224, 224))
            inputs = torchvision.transforms.ToTensor()(inputs)
            return inputs, label


    def countTypes(self):
        df = pandas.DataFrame(data=self.csv_data, columns=['id', 'label'])
        df['label']=df['label'].astype(int)
        df.hist(column=['label'])
        plt.show()
        return 

    def __len__(self):
        return len(self.csv_data)

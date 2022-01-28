from ctypes import resize
import torch
from PIL import Image

#TODO: Move constants to constants.py
INPUT_WIDTH=200
INPUT_HEIGHT=150

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, pictures, labels):
        self.pictures = pictures
        self.labels = labels

    def __getitem__(self, index):
        # Grab a single training example
        picture = self.pictures[index]
        # Load and resize the desired image
        im = self.resizeImage(picture)
        label = self.labels[index]
        example = (im, label)

        return example

    def __len__(self):
        return len(self.pictures)

    def resizeImage(self, im_path):
        im = Image.open(im_path)
        im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
        # im.show()
        return im

if __name__ == "__main__":
    datasetInstance = StartingDataset(["data/cassava-leaf-disease-classification/train_images/1000015157.jpg"], [0])
    # datasetInstance.resizeImage(datasetInstance.pictures[0])
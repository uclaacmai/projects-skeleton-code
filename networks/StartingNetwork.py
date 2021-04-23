import torch
import torch.nn as nn

#600x800

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x

 

 class CNN(nn.module):
    """
    Basic CNN to pass the data through
    """
    def __init__(self, input_channels, output_dim):
        super().__init__()

        #filter is 5, output channels is 6 (both can be changed)
        self.conv1 = nn.Conv2d(input_channels, 6, 5) 

        #the filter dimmensions of the pooling layer (here 2x2, can be changed)
        self.pool = nn.MaxPool2d(2, 2)

        #16 output channels

        #16 output channels, filter still at 5
        self.conv2 = nn.Conv2d (6, 16, 5)

        #16 channels, not sure about 4x4
        self.fc = FCNetwork(16 * 4 * 4, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.reshape(x, (-1, 5 * 5 * 5))
        x = self.fc(x)
        return x


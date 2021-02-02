import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
       # print(x.shape)
        x = self.fc(x)
       # print(x.shape)
        x = self.sigmoid(x)
        #print(x.shape)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self, input_channels, output_dim):  #input is a 32 * 3 * 800 * 600 tensor
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.first = StartingNetwork(16*197*147, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.first.forward(x)
        return x
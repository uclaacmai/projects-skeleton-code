import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super(StartingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(25088,128)
        self.fc2 = nn.Linear(128,output_dim)
        
    def forward(self, x):
        print('Beginning shape: ', x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        print('Shape after CNN: ', x.shape)
        x = torch.reshape(x,[32, -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

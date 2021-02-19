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
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.flatten(x)
        #print(x.shape)
        x = F.relu(self.fc1(x))        
        x = (self.fc2(x))
        #print(x.shape)
        #x = self.sigmoid(x)
        #print(x.shape)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self, input_channels, output_dim):  #input is a 32 * 3 * 800 * 600 tensor
        super().__init__()
        #self.conv1 = nn.Conv2d(input_channels, out_channels=6, kernel_size=5)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.first = StartingNetwork(512, output_dim) #16*197*147, output_dim)
        #resnet18:
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.resnet.eval()

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        with torch.no_grad():
            x = self.resnet(x)
        x = torch.squeeze(x)
        x = self.first.forward(x)
        return x
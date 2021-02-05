import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super(StartingNetwork, self).__init__()
        self.squeezenet = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        for param in self.squeezenet.parameters():
            param.requires_grad = False
        # self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        # # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(256, 10, kernel_size=3, stride=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_size = 1000
        # 10 * 35 *48
        # 33600
        # 10*50*37
        self.fc1 = nn.Linear(self.flatten_size,128)
        self.fc2 = nn.Linear(128,output_dim)
        
    def forward(self, x):
        # x = torch.reshape(x,[-1,3,600,800])
        # print('Beginning shape: ', x.shape)
        # x = self.conv1(x)
        # # print("finished conv1",x.shape)
        # x = F.relu(x)
        # # print("finished relu",x.shape)
        # x = self.maxpool1(x)
        # # print("finished maxpool",x.shape)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.maxpool1(x)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.maxpool1(x)
        # x = self.conv4(x)
        # # x = F.relu(x)
        # # x = self.maxpool1(x)
        # # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.maxpool1(x)
        # # print('Shape after CNN: ', x.shape)
        x = self.squeezenet(x)
        # print(x.shape)
        x = torch.reshape(x,[-1, self.flatten_size])
        # print('After reshaping',x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # print("SHape after FCN",x.shape)
        return x

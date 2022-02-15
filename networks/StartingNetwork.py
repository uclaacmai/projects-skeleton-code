import torch
import torch.nn as nn
import torch.nn.functional as F

#RuntimeError: mat1 and mat2 shapes cannot be multiplied (16384x1 and 1000x512)
class StartingNetwork(nn.Module):
    def __init__(self):
        # Call nn.Module's constructor
        super().__init__()
        
        # Transfer resnet model (do later after data augmentation, batch normalization)
        self.model_a = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # Remove last layer
        self.model_a = torch.nn.Sequential(*(list(self.model_a.children())[:-1]))
        # Output of ResNet: 1000-d tensor
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)

        # # 150 x 200 x 3
        # self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)

        # self.bn1 = nn.BatchNorm2d(6)

        # # 150 x 200 x 6
        # self.pool1 = nn.MaxPool2d(2, 2)

        # # 75 x 100 x 6
        # self.conv2 = nn.Conv2d(6, 8, kernel_size=5, padding=0)

        # self.bn2 = nn.BatchNorm2d(8)


        # # 71 x 96 x 8
        # self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # # 36 x 48 x 8
        # self.fc1 = nn.Linear(36 * 48 * 8, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # Forward propagation
        with torch.no_grad():
            x = self.model_a(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        # x = self.bn1(self.conv1(x))
        # x = F.relu(x)
        # x = self.pool1(x)
        # x = self.bn2(self.conv2(x))
        # x = F.relu(x)
        # x = self.pool2(x)

        # x = torch.reshape(x, (-1, 36 * 48 * 8))
        # x = self.fc1(x)
        # x = F.relu(x)

        # x = self.fc2(x)
        # x = F.relu(x)

        # x = self.fc3(x)
        
        return x
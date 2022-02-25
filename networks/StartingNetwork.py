import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(nn.Module):
    def __init__(self):
        # Call nn.Module's constructor
        super().__init__()
        
        # Transfer resnet model (do later after data augmentation, batch normalization)
        self.model_a = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # Remove last layer
        self.model_a = torch.nn.Sequential(*(list(self.model_a.children())[:-1]))
        # Output of ResNet: 512-d tensor
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # Forward propagation
        with torch.no_grad():
            x = self.model_a(x)

        x = x.reshape((32, -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
import torch
import torch.nn as nn
import torchvision.models as models


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224 * 224 * 3, 5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x


class Model_b(nn.Module):
    def __init__(self):
        super(Model_b, self).__init__()
        self.encoder = models.resnet18(pretrained = True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        features = torch.flatten(features, 1)
        return self.fc(features)






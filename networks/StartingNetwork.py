import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(nn.Module):
    def __init__(self):
        # Call nn.Module's constructor
        super().__init__()
        
        # 150 x 200 x 3
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)

        # 150 x 200 x 6
        self.pool1 = nn.MaxPool2d(2, 2)

        # 75 x 100 x 6
        self.conv2 = nn.Conv2d(6, 8, kernel_size=5, padding=0)

        # 71 x 96 x 8
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # 36 x 48 x 8
        self.fc1 = nn.Linear(36 * 48 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # Forward propagation
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.reshape(x, (-1, 36 * 48 * 8))
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    model = StartingNetwork()
    test_im = torch.tensor(150, 200, 3)
    x = model.forward(test_im)
    print("")
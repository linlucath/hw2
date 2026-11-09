import torch
from torch import nn


class DiCNN(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(9, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, lms, pan):
        x = self.relu(self.conv1(torch.cat((lms, pan), dim=1)))
        return lms + self.conv3(self.relu(self.conv2(x)))

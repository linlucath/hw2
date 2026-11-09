from torch import nn


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv20 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.conv21(self.relu(self.conv20(x)))


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.res1, self.res2, self.res3, self.res4 = (ResBlock() for _ in range(4))
        # conv3 matches the official checkpoint's name.
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, lms, pan):
        x = self.relu(self.conv1(pan.repeat(1, 8, 1, 1) - lms))
        for block in (self.res1, self.res2, self.res3, self.res4):
            x = block(x)
        return lms + self.conv3(x)

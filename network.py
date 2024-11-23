import torch as th
from torch import nn

x = th.randn((4, 3, 64, 64))

class Maxout2D(nn.Module):
    def __init__(self, max_out):
        super().__init__()

        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x: th.Tensor):
        B, C, H, W = x.shape

        h = x.permute(0, 2, 3, 1).view(B, H * W, C)
        h = self.max_pool(h)
        h = h.permute(0, 2, 1).view(B, C // self.max_out, H, W).contiguous()

        return h

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=.85),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=.85),
        )

        self.downsample = None

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=.85),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: th.Tensor):
        y = x
        h = self.layers(x)

        if self.downsample:
            y = self.downsample(x)

        h += y
        h = self.relu(h)

        return h

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=1, padding="same"),
            nn.BatchNorm2d(num_features=10, momentum=.85),
            Maxout2D(2),
            nn.Conv2d(5, 6 , kernel_size=1, padding="same"),
            nn.BatchNorm2d(num_features=6, momentum=.85),
            Maxout2D(2),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=.85),
            nn.ReLU(),
            Block(64, 64),
            Block(64, 64),
            Block(64, 128),
            Block(128, 128),
            Block(128, 128),
            Block(128, 256, stride=2),
            Block(256, 256),
            Block(256, 256),
            Block(256, 512, stride=2),
            Block(512, 512),
            Block(512, 512),
            nn.AvgPool2d(7, stride=2),
            nn.Dropout(.5),
        )

        self.dense = nn.Linear(512, 24)

    def forward(self, x: th.Tensor):
        h = self.layers(x)
        h = h.view(x.shape[0], -1)
        h = self.dense(h)

        return h

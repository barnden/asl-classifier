import torch as th
from torch import nn

from layers import Maxout2D, Block

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=1, padding="same"),
            nn.BatchNorm2d(num_features=10),
            Maxout2D(2),
            nn.Conv2d(5, 6 , kernel_size=1, padding="same"),
            nn.BatchNorm2d(num_features=6),
            Maxout2D(2),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
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

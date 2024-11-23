import torch as th
from torch import nn

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
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.downsample = None

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
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

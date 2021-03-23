import torch
import torch.nn as nn


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
class MobileNetV1(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 depth_multiplier=1.0,
                 min_depth=8):
        super(MobileNetV1, self).__init__()
        depth = lambda d: max(int(d * depth_multiplier), min_depth)
        self.mobelnet = nn.Sequential(
            conv_bn(3, depth(32), 2),
            conv_dw(depth(32), depth(64), 1),
            conv_dw(depth(64), depth(128), 2),
            conv_dw(depth(128), depth(128), 1),
            conv_dw(depth(128), depth(256), 2),
            conv_dw(depth(256), depth(256), 1),
            conv_dw(depth(256), depth(512), 2),
            conv_dw(depth(512), depth(512), 1),
            conv_dw(depth(512), depth(512), 1),
            conv_dw(depth(512), depth(512), 1),
            conv_dw(depth(512), depth(512), 1),
            conv_dw(depth(512), depth(512), 1),
            conv_dw(depth(512), depth(1024), 2),
            conv_dw(depth(1024), depth(1024), 1),
            nn.AvgPool2d(7)
        )

        self.classifier = nn.Linear(depth(1024), num_classes)

    def forward(self, x):
        x = self.mobelnet(x)
        x = x.view(-1, 1024)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

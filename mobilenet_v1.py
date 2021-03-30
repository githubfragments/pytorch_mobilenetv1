import torch
import torch.nn as nn
from typing import List, Any


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
# https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
class MobileNetV1(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 depth_multiplier=1.0,
                 min_depth=8):
        super(MobileNetV1, self).__init__()
        def depth(d): return max(int(d * depth_multiplier), min_depth)
        features: List[nn.Module] = [conv_bn(3, depth(32), 2)]

        conv_dw_setting = [
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 1024, 2],
            [1024, 1024, 1]
        ]

        for in_c, out_c, stride in conv_dw_setting:
            features.append(conv_dw(depth(in_c), depth(out_c), stride))

        features.append(nn.AvgPool2d(7))
        # make it nn.Sequential
        self.mobilenet = nn.Sequential(*features)

        self.classifier = nn.Linear(depth(1024), num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v1(pretrained_weights=None, **kwargs: Any):
    model = MobileNetV1(**kwargs)
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights)
        model.load_state_dict(state_dict)
    return model


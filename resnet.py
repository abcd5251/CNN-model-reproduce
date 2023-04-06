import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 4
features = torch.randn(batch_size, 32, 32, 32)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class InceptionResBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch1 = BasicConv2d(in_planes, 32, (1, 1), 1, 0)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, 32, (1, 1), 1),
            BasicConv2d(32, 32, (3, 3), 1, 1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes, 32, (1, 1), 1),
            BasicConv2d(32, 48, (3, 3), 1, 1),
            BasicConv2d(48, 64, (3, 3), 1, 1)
        )
        self.branch4 = BasicConv2d(128, 384, (1, 1), stride, 0)

        if (in_planes != out_planes) or (stride != 1):
            self.map = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
        else:
            self.map = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        identity = self.map(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x1_3 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.branch4(x1_3)

        x = identity + x4

        output = self.relu(x)
        return output 

model = InceptionResBlock(32, 384)
output = model(features)
print(f"Input shape : {features.shape}\nOutput shape : {output.shape}")
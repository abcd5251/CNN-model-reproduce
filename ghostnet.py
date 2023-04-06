import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=1, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup      # output channel
        init_channels = math.ceil(oup / ratio)  # original pointwise convolution 
        new_channels = init_channels*(ratio-1) # keep identity channels of pointwise convolution

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size,
                      stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        ) ## 1*1 depthwise convolution

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1) # combine identity and new one 
        return out[:, :self.oup, :, :]
    
model = GhostModule(64, 64, 3)
batch_size = 4
features = torch.randn(batch_size, 64, 32, 32)
output = model(features)
print(f"Input shape : {features.shape}\nOutput shape : {output.shape}")
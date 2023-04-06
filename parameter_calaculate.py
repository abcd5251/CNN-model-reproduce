import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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

    
model = nn.Conv2d(5, 12, kernel_size=(3, 3), stride=1, padding=0, bias=True)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"params : {params}")
print(f"calculation : {((3 * 3) * 5 + 1) * 12}")

# ((kernel size * kernel size) * (input channel) + bias) * (output channel)
# (3 * 3 * 5(input channel) + 1(bias) * 12(output channel)

print("\n")


model = BasicConv2d(12, 12, 3, 1, 1) # (input channel, output channel, kernelsize, stride, padding)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"params : {params}")
print(f"calculation : {((3 * 3) * 12 + 1 ) * 12 + 12 * 2}")
# ((kernel size * kernel size) * (input channel) + bias) * (output channel) + BN neurons * 2(shift and scale parameters)

summary(model, input_size = (12, 64, 64), batch_size = -1)


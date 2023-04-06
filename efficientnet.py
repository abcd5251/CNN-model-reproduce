import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v: float, divisor: int, min_value=None) -> int: # To make sure it can be divided
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SiLU(nn.Module):  #The SiLU function is also known as the swish function - export-friendly version of nn.SiLU() 
    def forward(self, x): 
        return x * torch.sigmoid(x) # because memory usage too big 
    
# memory efficient version (can reduce almost 20 - 30% memory usage)
sigmoid = torch.nn.Sigmoid()
class SwishCustomized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = SwishCustomized.apply

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input, True)
        return scale * input
    
class MBConv(nn.Module):
    def __init__(self, nin, nout, stride, expansion_ratio=4, se=True):
        super(MBConv, self).__init__()
        expand_ = expansion_ratio * nout
        self.stride = stride
        self.se = se
        self.expansion = nn.Sequential(nn.Conv2d(nin, expand_, kernel_size=1), 
                                       nn.BatchNorm2d(expand_), Swish())
        
        self.depthwise = nn.Sequential(nn.Conv2d(expand_, expand_, kernel_size=3, padding=1, stride=stride, groups=nin), 
                                       nn.BatchNorm2d(expand_), Swish())
        
        self.pointwise = nn.Sequential(nn.Conv2d(expand_, nout, kernel_size=1), 
                                       nn.BatchNorm2d(nout))
        
        if self.se:
            self.squeeze_excitation = SqueezeExcitation(expand_)
        
        if stride == 1:
            if (nin != nout):
                self.map = nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
            else:
                self.map = nn.Identity()
    def forward(self, x):
        out = self.expansion(x)
        out = self.depthwise(out)
        
        if self.se:
            out = self.squeeze_excitation(out)
            
        out = self.pointwise(out)
            
        if self.stride == 1:
            identity = self.map(x)
            out += identity
        return out
    
model = MBConv(64, 64, 1, 6, se=True)
batch_size = 4
features = torch.randn(batch_size, 64, 32, 32)
output = model(features)
print(f"Input shape : {features.shape}\nOutput shape : {output.shape}")


# efficientV2 : accuracy may not higher than V1, but faster than V1, sometimes accuracy higher because of NAS
class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input, True)
        return scale * input
    
class FusedMBConv(nn.Module):
    def __init__(self, nin, nout, stride, expansion_ratio=4, se=True):
        super(FusedMBConv, self).__init__()
        expand_ = expansion_ratio * nout
        self.stride = stride
        self.se = se
        self.expansion = nn.Sequential(nn.Conv2d(nin, expand_, kernel_size=3, padding=1), 
                                       nn.BatchNorm2d(expand_), SiLU())
        
        if self.se:
            self.squeeze_excitation = SqueezeExcitation(expand_)
            
        self.pointwise = nn.Sequential(nn.Conv2d(expand_, nout, kernel_size=1), 
                                       nn.BatchNorm2d(nout))
        
        
        if stride == 1:
            if (nin != nout):
                self.map = nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
            else:
                self.map = nn.Identity()
    def forward(self, x):
        out = self.expansion(x)
        
        if self.se:
            out = self.squeeze_excitation(out)
            
        out = self.pointwise(out)
            
        if self.stride == 1:
            identity = self.map(x)
            out += identity
        return out

model = FusedMBConv(64, 64, 1)   
batch_size = 4
features = torch.randn(batch_size, 64, 32, 32)
output = model(features)
print(f"Input shape : {features.shape}\nOutput shape : {output.shape}")

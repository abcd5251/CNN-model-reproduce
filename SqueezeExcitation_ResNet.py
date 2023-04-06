import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SqueezeExcitation(nn.Module):
   
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8) # T / k 
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
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
    
    
class SEResidualBlockV2(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, se=True):
        super().__init__()
        self.se = se
        squeeze = out_planes // 4
        self.conv1 = BasicConv2d(in_planes, squeeze, (1, 1), 1, 0) ## squeeze
        self.conv2 = BasicConv2d(squeeze, squeeze, (3, 3), stride=stride, padding=1) ## squeeze
        self.conv3 = nn.Conv2d(squeeze, out_planes, (1, 1), 1, 0) ## expand
        
        if self.se:
            self.squeeze_excitation = SqueezeExcitation(out_planes)
        if (in_planes != out_planes) or (stride != 1): #if stride = 2, feature map will be downsampling, make sure dimension is same
            self.map = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
        else:
            self.map = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se:
            x = self.squeeze_excitation(x)
        x += identity
        return x 
    
model = SEResidualBlockV2(64, 64, se=True)
batch_size = 4
features = torch.randn(batch_size, 64, 32, 32)
output = model(features)
print(f"Input shape : {features.shape}\nOutput shape : {output.shape}")


demo = False  # use to prove that Conv2d 1 channel is same like Linear(fully connected)
if demo:
    model = nn.Conv2d(10, 10, 1) # input 10 feature map , output 10 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"params : {params}")

    model = nn.Linear(10, 10)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"params : {params}")
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

class DepthwiseSeparable(nn.Module): # Mobile CNN
    def __init__(self, nin, nout):
        super(DepthwiseSeparable, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin), # groups means depthwise convolution, inputchannel 12, groups 4, feature map 3 can see
                                       nn.BatchNorm2d(nin), nn.ReLU6(inplace=True))
        self.pointwise = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=1), # input channel, output channel, kernel size (1,1)
                                       nn.BatchNorm2d(nout), nn.ReLU6(inplace=True))
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class General(nn.Module): # normal CNN
    def __init__(self, nin, nout):
        super(General, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(nin, nin, kernel_size=3, padding=1), 
                                       nn.BatchNorm2d(nin), nn.ReLU6(inplace=True))
        self.pointwise = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=1), 
                                       nn.BatchNorm2d(nout), nn.ReLU6(inplace=True))
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
# nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
# ( 3 * 3 * 32 + 1 ) * 32  original 
# (3 * 3 * 1 + 1) * 32   groups, ignore channelwise dimension groups 32 / 32 = 1
# 2(every neuron have shift and vairance)* 32(neuron), nn.BatchNorm2d(32) for batchnormalization
# ReLU no need of parameters
# (3 * 3 * 1 + 1) * 32 + 2 * 32 + (1 * 1 * 32 +1) * 32 + 2 * 32,  depthwise + pointwise

model = DepthwiseSeparable(32, 32)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"DepthwiseSeparable params : {params}")

model = General(32, 32)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"General params : {params}")

# MobileNet V2
# expand t times on depthwise convolution 

class InvertedResidual(nn.Module):
    def __init__(self, nin, nout, stride, expansion_ratio=4):
        super(InvertedResidual, self).__init__()
        expand_ = expansion_ratio * nin # tk , channel expand 4 times
        self.stride = stride
        self.expansion = nn.Sequential(nn.Conv2d(nin, expand_, (1,1)), nn.BatchNorm2d(expand_),  # expansion 
                                       nn.LeakyReLU(inplace = True)) # inplace true means override last layer vector,parameter to reduce memory, if backpropagation goes worng then set it to false.  
        self.depthwise = nn.Sequential(nn.Conv2d(expand_, expand_, kernel_size=3, padding=1, groups=expand_), # groups means depthwise convolution, inputchannel 12, groups 4, feature map 3 can see
                                       nn.BatchNorm2d(expand_), nn.ReLU6(inplace=True))
        self.pointwise = nn.Sequential(nn.Conv2d(expand_, nout, kernel_size=1), # kernel size 1 * 1
                                       nn.BatchNorm2d(nout), nn.ReLU6(inplace=True))
        
        if stride == 1: 
            if (nin != nout):
                self.map = nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
            else: # let output channel same 
                self.map = nn.Identity()
    def forward(self, x):
       
        out = self.expansion(x)
        out = self.depthwise(out)
        out = self.pointwise(out)
        
        if self.stride == 1:
            identity = self.map(x) 
            out += identity
        return out

batch_size = 4 
features = torch.randn(batch_size, 24, 32, 32) # batchsize, inputchannel, height, width
model = InvertedResidual(24, 96, 1)
output = model(features)
print(output.shape)

# MobileNetV3 

class HSigmoid(nn.Module):
    def forward(self, x):
        return (F.relu6(x + 3, inplace=True) / 6)
            
class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsigmoid = HSigmoid()
    def forward(self, x):
        return x * self.hsigmoid(x)
    
h_sigmoid = HSigmoid()
h_swish = HSwish()

class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.swish = HSwish()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.hsigmoid = HSigmoid()

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.swish(scale)
        scale = self.fc2(scale)
        return self.hsigmoid(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input, True)
        return scale * input
    
class MBConv(nn.Module): # Inverted residule + SE block 
    def __init__(self, nin, nout, stride, expansion_ratio=4, se=True):
        super(MBConv, self).__init__()
        expand_ = expansion_ratio * nout
        self.stride = stride
        self.se = se
        self.expansion = nn.Sequential(nn.Conv2d(nin, expand_, kernel_size=1), 
                                       nn.BatchNorm2d(expand_), HSwish())
        
        self.depthwise = nn.Sequential(nn.Conv2d(expand_, expand_, kernel_size=3, padding=1, stride=stride, groups=nin), 
                                       nn.BatchNorm2d(expand_), HSwish())
        
        self.pointwise = nn.Sequential(nn.Conv2d(expand_, nout, kernel_size=1), 
                                       nn.BatchNorm2d(nout))
        
        if self.se:
            self.squeeze_excitation = SqueezeExcitation(expand_)
        
        if stride == 1: # in every ResNet can see, make sure channel is same 
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

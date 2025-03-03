"""Contains definitions for VGG16 and Modified ResNet-18 networks. Adapted in part from https://github.com/arunmallya/packnet/blob/master/src/networks.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from functools import partial
from typing import Any, cast, Dict, List, Optional, Union




 
########################################################################################################################
#Modified Resnet
#########################################################################################################################################################
#!# Modified ResNet where all residual connections are 1x1 conv layers with simple batchnorm

# Source: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

"""3x3 convolution with padding"""
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self,inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        ### Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        ### All bn layers have affine and track_running_stats = False, otherwirse issues arise when suddenly sharing different sets of weights
        self.bn1 = norm_layer(planes, affine=False, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=False, track_running_stats=False)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out1 = self.bn1(out)
        out2 = self.relu1(out1)

        out3 = self.conv2(out2)
        out4 = self.bn2(out3)

        identity = self.downsample(x)

        out5 = out4 + identity
        out6 = self.relu2(out5)

        return out6





class ModifiedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, nf=64, zero_init_residual=False, norm_layer=None):
        # print("Change updates")
        super(ModifiedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block

        self.inplanes = nf


        # 3x32x32 images: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf, layers[0])
        self.layer2 = self._make_layer(block, nf*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, nf*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, nf*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        self.shared = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool)
        self.classifier = None

        #!# Removed intialization of bn layers, since they no longer have parameters to set
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        ### if stride != 1 or self.inplanes != planes:
        downsample1 = nn.Sequential(
            conv1x1(self.inplanes, planes, stride),
            norm_layer(planes, affine=False, track_running_stats=False),
        )
        #!# The second skip junction in layers 2-4 normally dont downsample, so manually setting the stride to 1 to maintain the same feature map shape as in resnet18
        downsample2 = nn.Sequential(
            conv1x1(planes, planes, stride=1),
            norm_layer(planes, affine=False, track_running_stats=False),
        )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample1,
                norm_layer,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    downsample = downsample2,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.shared[0](x)
        x = self.shared[1](x)
        x = self.shared[2](x)
        x = self.shared[3](x)

        x = self.shared[4](x)
        x = self.shared[5](x)
        x = self.shared[6](x)
        x = self.shared[7](x)

        x = self.shared[8](x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x



def modifiedresnet18(num_classes=10, nf=64, **kwargs):
    return ModifiedResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, nf=nf, **kwargs)







#########################################################################################################################################################
# VGG16 
#########################################################################################################################################################
### Adapted in part from https://github.com/arunmallya/packnet/blob/master/src/networks.py

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = nn.Linear(512, 4096, bias=False) 
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(4096, 4096, bias=False) 
        self.relu2 = nn.ReLU(True)
        features = list(features.children())
        features.extend([
            self.avgpool,
            View(-1, 512),
            self.linear1,
            self.relu1,
            self.linear2,
            self.relu2,
        ])

        ### All layers except for the classifier are shared between tasks (with masking)
        self.shared = nn.Sequential(*features)

        self.classifier = None

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    # nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=False, track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



 #          1   4         8    11       15    18   21        25  28    31        35   38   41   

cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


def vgg16(cfg: str = "D", batch_norm: bool = True, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model
    
    














# This file is not used, it exists for when the model needs to split across GPUs. 
# Model parallel requires DeepPool to have a fine-grain understanding of the layers of the model,
# and below we see what would be required for DeepPool to get that.

import torch
import torch.nn as nn
from parallelizationPlanner import CostSim

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet34(nn.Module):

    def __init__(self, deeppool: CostSim, num_classes=1000):
        super().__init__()
        
        layers=[3, 4, 6, 3]

        self.inplanes = 64

        self.conv1 = deeppool.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = deeppool.BatchNorm2d(self.inplanes)

        self.relu = deeppool.ReLU()
        self.maxpool = deeppool.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        deeppool.GeneralLayer(self.layer1, "layer1", {}, mustTrace=True)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        deeppool.GeneralLayer(self.layer2, "layer2", {}, mustTrace=True)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        deeppool.GeneralLayer(self.layer3, "layer3", {}, mustTrace=True)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        deeppool.GeneralLayer(self.layer4, "layer4", {}, mustTrace=True)

        self.avgpool = deeppool.AdaptiveAvgPool2d((1, 1))
        self.flatten = deeppool.Flatten()
        self.fc = deeppool.Linear(512 , num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = self.flatten(x)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x
import torch
import torch.nn as nn
import sys
import os
import time

from typing import Dict, Type, Any, Callable, Union, List, Optional
# from .utils import load_state_dict_from_url

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# sys.path.append('.')
from parallelizationPlanner import CostSim
from parallelizationPlanner import GpuProfiler
from clusterClient import ClusterClient
from jobDescription import TrainingJob


import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torch import Tensor

# from .utils import (
#     # round_filters,
#     # round_repeats,
#     # drop_connect,
#     # get_same_padding_conv2d,
#     get_same_padding_conv_transpose2d,
#     # get_model_params,
#     # efficientnet_params,
#     # load_pretrained_weights,
#     Swish,
#     MemoryEfficientSwish,
# )

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_same_padding_conv_transpose2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return ConvTranspose2dDynamicSamePadding
    else:
        return partial(ConvTranspose2dStaticSamePadding, image_size=image_size)

# class Conv2dDynamicSamePadding(nn.Conv2d):
#     """ 2D Convolutions like TensorFlow, for a dynamic image size """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
#         super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#         self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

#     def forward(self, x):
#         ih, iw = x.size()[-2:]
#         kh, kw = self.weight.size()[-2:]
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
#         return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# class Conv2dStaticSamePadding(nn.Conv2d):
#     """ 2D Convolutions like TensorFlow, for a fixed image size"""

#     def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
#         super().__init__(in_channels, out_channels, kernel_size, **kwargs)
#         self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

#         # Calculate padding based on image size and save it
#         assert image_size is not None
#         ih, iw = image_size if type(image_size) == list else [image_size, image_size]
#         kh, kw = self.weight.size()[-2:]
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
#         else:
#             self.static_padding = Identity()

#     def forward(self, x):
#         x = self.static_padding(x)
#         x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return x

class ConvTranspose2dDynamicSamePadding(nn.ConvTranspose2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            #x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            #x = F.pad(x, [2, 2,2, 2])
            x = F.pad(x, [0 , 0,0, 0])
        #return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        if ih < 20:
            return F.conv_transpose2d(x, self.weight, bias=self.bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups) 
        else:
            return F.conv_transpose2d(x, self.weight, bias=self.bias, stride=self.stride, padding=1, dilation=self.dilation, groups=self.groups)

class ConvTranspose2dStaticSamePadding(nn.ConvTranspose2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, custom_previous_layers: list = None) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return cs.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     custom_previous_layers=custom_previous_layers)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, custom_previous_layers: list = None) -> nn.Conv2d:
    """1x1 convolution"""
    return cs.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, custom_previous_layers=custom_previous_layers)

class AddLayer(nn.Module):
    def __init__(self, t1, t2):
        super(AddLayer, self).__init__()
        self.t1 = t1
        self.t2 = t2

    def forward(self, x, y):
        return x+y

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = cs.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        identity = cs.layers[-1]
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = cs.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        pre_downsample = cs.layers[-1]

        if downsample is not None:
            # join_layers += [cs.layers[-1]]
            # self.addLayer = AddLayer(join_layers[0], join_layers[1])
            # print(join_layers[0].name, join_layers[1].name)
            # cs.GeneralLayer(self.addLayer, "add", {}, mustTrace=True, custom_previous_layers = join_layers)
            self.downsample = downsample([identity])
            identity = cs.layers[-1]
        else:
            self.downsample = None

        self.addLayer = AddLayer(None, None)
        cs.GeneralLayer(self.addLayer, "add", {}, mustTrace=True, custom_previous_layers = [pre_downsample, identity])
        self.relu2 = cs.ReLU(inplace=False)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        # downsample: Optional[nn.Module] = None,
        downsampleParams: Optional[tuple] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = cs.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        layerSideBranch = cs.layers[-1]
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = cs.BatchNorm2d(width)
        self.relu1 = cs.ReLU(inplace=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = cs.BatchNorm2d(width)
        self.relu2 = cs.ReLU(inplace=False)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = cs.BatchNorm2d(planes * self.expansion)
        layerMainBranch = cs.layers[-1]
        if False and downsampleParams is not None: # hack to test resnet152
            convDownsample = conv1x1(*(downsampleParams[0]), [layerSideBranch])
            layerSideBranch = cs.layers[-1]
            self.downsample = nn.Sequential(convDownsample, norm_layer((downsampleParams[1])))
        else:
            self.downsample = None
        # self.relu = cs.ReLU(inplace=True, custom_previous_layers=[layerMainBranch, layerSideBranch])
        self.relu3 = cs.ReLU(inplace=False)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels: int = 2,
        global_params: Dict = None
    ) -> None:
    # def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None, in_channels=3, global_params=None):
        super(ResNet, self).__init__()
        print(global_params)
        self._global_params = global_params

        self.printcount = 0
        if norm_layer is None:
            norm_layer = cs.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # self.groups = groups
        # self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # # self.conv1_new = nn.Conv2d(64, self.inplanes, kernel_size=3, stride=1, padding=1,
        # #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = cs.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = cs.BatchNorm2d(self.inplanes)
        self.relu1 = cs.ReLU(inplace=False)
        self.maxpool = cs.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = cs.AdaptiveAvgPool2d((1, 1))
        # cs.Flatten()
        # self.fc = cs.Linear(512 * block.expansion, num_classes)



        # Batch norm parameters
        bn_mom = 0.99 #elf._global_params['batch_norm_momentum']
        bn_eps = 1e-3 #self._global_params['batch_norm_epsilon']

        # Head
        # ConvTranspose2d = get_same_padding_conv_transpose2d()
        in_channels = 256*block.expansion  # output of final block
        #scaled_output_filters = unscaled_output_filters*1280/self._blocks_args[-1].output_filters
        #out_channels = round_filters(scaled_output_filters, self._global_params)
        out_channels = 3
        self.conv_head0 = cs.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=(2,2), padding=0, bias=False)
        
        self.bn_head0 = cs.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.swish = Swish()
        cs.GeneralLayer(self.swish, "swish", {}, mustTrace=True)
        self.conv_head1 = cs.ConvTranspose2d(out_channels, 3,
                                           kernel_size=3, stride=(2,2), bias=False, padding=1, dilation=(1,1))
        # self._bn1 = cs.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # self._conv_head2 = cs.ConvTranspose2d(out_channels, 3,
        #                                    kernel_size=3, stride=(2,2), bias=False)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, custom_previous_layers=prevlayer),
            #     cs.BatchNorm2d(planes * block.expansion),
            # )
            downsample = lambda prevlayer: nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, custom_previous_layers=prevlayer),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # def set_swish(self, memory_efficient=True):
    #     """Sets swish function as memory efficient (for training) or standard (for export)"""
    #     self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
    #     # for block in self._blocks:
    #     #     block.set_swish(memory_efficient)

    def _space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

    def _depth_to_space(self, x, block_size):
        n, c, h, w = x.size()
        return x.view(n, c // (block_size ** 2), h * block_size, w * block_size)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # x = self._space_to_depth(x, 8)
        # x = self.conv1_new(x)
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # print(x.size())
        x = self.layer1(x)
        # print('layer1', x.size())
        x = self.layer2(x)
        # print('layer2', x.size())
        x = self.layer3(x)
        # x = self._depth_to_space(x, 2)
        # print('layer3', x.size())
        # x = self.layer4(x)

        # print(x.size())
        x = self.conv_head0(x)
        # print('_conv_head0', x.size())
        x = self.swish(self.bn_head0(x))
        # print('_swish', x.size())
        x = self.conv_head1(x)
        # print('_conv_head1', x.size())
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, global_params=None, **kwargs):
    model = ResNet(block, layers, global_params=global_params, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        weights_dict = torch.load(pretrained)
        del weights_dict['conv1.weight']
        # del weights_dict['conv1.bias']
        model.load_state_dict(weights_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, global_params=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print(global_params)
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, global_params,
                   **kwargs)


def resnet34(pretrained=False, progress=True, global_params=None, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, global_params,
                   **kwargs)


def resnet50(pretrained=False, progress=True, global_params=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 9, 3], pretrained, progress, global_params,
                   **kwargs)


def resnet101(pretrained=False, progress=True, global_params=None, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, global_params,
                   **kwargs)


def resnet152(pretrained=False, progress=True, global_params=None, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, global_params,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)






def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
    # profiler = GpuProfiler("cuda")
    # profiler.loadProfile()
    global cs
    cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="profile/A100_anvil.prof")#, gpuProfileLocSub="resnetLayerGpuProfileA100.txt")
    model = resnet34(pretrained='/anvil/data/pretrained/resnet34.pth')
    # model.load_state_dict(torch.load('/anvil/planes_checkpoints_0_997AUC/weights_0499.pth'), strict=True)
    # model.load_state_dict(torch.load('/DeepPool/weights_0499.pth'), strict=True)
    # model = resnet152()
    # model = wide_resnet101_2()
    # cs.printAllLayers(slient=True)
    cs.printAllLayers(slient=False)
    cs.computeInputDimensions((2,240,240))
    cs.to_dot("Digraph", globalBatch, justdag=True)

    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit, lossfn=0)
    # job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, lossfn=2, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    job, iterMs, gpuMs, maxGpusUsed = cs.JustDoDP(gpuCount, globalBatch, lossfn=2)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))

    jobInJson = job.dumpInJSON()
    # profiler.saveProfile()
    # for rank in range(4):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    # job2 = TrainingJob("test", None, None, 0, 0, "")
    # job2.loadJSON(jobInJson)
    # assert(jobInJson == job2.dumpInJSON())
    # print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "anvil_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, use_be)

    if simResultFilename != None:
        f = open(simResultFilename, "a")
        f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, gpuCount, iterMs, gpuMs))
        f.close()

        if gpuCount == 8:
            f = open(simResultFilename, "r")
            print(f.read())
            f.close()


def runAllConfigs(modelName: str, clusterType: str, simOnly=True):
    if clusterType == "V100":
        netBw = 22937
    elif clusterType == "A100":
        netBw = 2.66E5
    elif clusterType == "B100":
        netBw = 2.66E5 * 5
    else:
        print("Wrong cluster type. Put either V100 or A100")

    gpuCounts = [1, 2, 4, 8, 16]
    # gpuCounts = [1, 2, 4]
    globalBatchSize = 128
    # globalBatchSize = 16
    # globalBatchSize = 8
    limitAndBaseline = [(2.0, True, False), (99, False, False), (1.5, False, False), (2.0, False, False), (2.5, False, False)]
    # limitAndBaseline = [(99, False, True)]
    # limitAndBaseline = []
    for lim, baseline, spatialSplit in limitAndBaseline:
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim)
        f = open(simResultFilename, "w")
        f.write("#batch GPUs IterMs  GpuMs\n")
        f.close()

        for gpuCount in gpuCounts:
            if not simOnly:
                preSize = os.stat('runtimeResult.data').st_size
            main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename, simOnly=simOnly)
            # check exp finished.
            if not simOnly:
                print("runtimeResult.data's original size: ", preSize)
                while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
                    time.sleep(10)
                print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
        if not spatialSplit and not simOnly:
            fw = open("%s_%s_b%d_lim%2.1f_run.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim), "w")
            fr = open('runtimeResult.data', "r")
            fw.write("#batch GPUs IterMs  GpuMs\n")
            fw.write(fr.read())
            fw.close()
            fr.close()

        fr = open('runtimeResult.data', "w")
        fr.close()

    # #################################
    # ## Profiling by batch size.
    # #################################
    # globalBatchSizes = [1,2,4,8,16,32,64,128]
    # lim, baseline, spatialSplit = (2.0, True, False)
    # simResultFilename = "%s_%s_varyBatch_sim.data" % (modelName, "DP" if baseline else "MP")
    # f = open(simResultFilename, "w")
    # f.write("#batch GPUs IterMs  GpuMs\n")
    # f.close()

    # # for gpuCount in gpuCounts:
    # gpuCount = 1
    # for globalBatchSize in globalBatchSizes:
    #     preSize = os.stat('runtimeResult.data').st_size
    #     main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename)
    #     # check exp finished.
    #     print("runtimeResult.data's original size: ", preSize)
    #     while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
    #         time.sleep(10)
    #     print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
    
    # if not spatialSplit:
    #     fw = open("%s_%s_varyBatch_run.data" % (modelName, "DP" if baseline else "MP"), "w")
    #     fr = open('runtimeResult.data', "r")
    #     fw.write("#batch GPUs IterMs  GpuMs\n")
    #     fw.write(fr.read())
    #     fw.close()
    #     fr.close()

    # fr = open('runtimeResult.data', "w")
    # fr.close()

def runStrongScalingBench(modelName='resnet50'):
    profiler = GpuProfiler("cuda")
    global cs
    netBw = 2.66E5
    cs = CostSim(profiler, netBw=netBw, verbose=False)
    inputSize = (3,224,224)
    if modelName == 'resnet50':
        model = resnet50(pretrained=False)
    elif modelName == 'resnet34':
        model = resnet34(pretrained=False)
    
    print("Model: ", modelName)
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 9)]:
        iterTime, fpTime, bpTime = profiler.benchModel(model, inputSize, batchSize)
        print(" %8d  %6.1f  %6.1f  %6.1f" %
            (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))

def generateJit():
    global cs
    netBw = 2.66E5
    cs = CostSim(GpuProfiler("cuda"), netBw=netBw, verbose=False)

    fakeInputSize = (16,3,224,224)
    fakeInput = torch.zeros(fakeInputSize)

    model = resnet152()
    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "beModules/resnet152.jit")
    
    model = resnext101_32x8d()
    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "beModules/resnext101_32x8d.jit")

    model = wide_resnet101_2()
    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "beModules/wide_resnet101_2.jit")
    

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True)
    elif len(sys.argv) >= 4:
        use_be = len(sys.argv) > 4 and int(sys.argv[4]) == 1
        if sys.argv[3] == "DP":
            main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True, use_be=use_be)
        else:
            main(int(sys.argv[1]), int(sys.argv[2]), amplificationLimit=float(sys.argv[3]), use_be=use_be)
    elif len(sys.argv) == 2:
        print("Run all configs")
        runAllConfigs("anvil", sys.argv[1])
    elif len(sys.argv) == 1:
        generateJit()
        # for modelName in ['resnet50', 'resnet34']:
        #     runStrongScalingBench(modelName)
    else:
        print("Wrong number of arguments.\nUsage: ")

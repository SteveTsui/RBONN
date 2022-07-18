import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timm.models.registry import register_model
import math
import torch
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd
import collections
from itertools import repeat

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)

        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class Bilinear_binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale_factor):
        
        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * scale_factor

        ctx.save_for_backward(weight, scale_factor)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, scale_factor = ctx.saved_tensors
        
        para_loss = 1e-4

        bin = 0.02

        weight_bin = torch.sign(weight) * bin
        
        gradweight = para_loss * (weight - weight_bin * scale_factor) + (gradOutput * scale_factor)
        
        grad_scale_1 = torch.sum(torch.sum(torch.sum(gradOutput * weight,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)
        
        grad_scale_2 = torch.sum(torch.sum(torch.sum((weight - weight_bin * scale_factor) * weight_bin ,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)

        gradscale = grad_scale_1 - para_loss * grad_scale_2

        return gradweight, gradscale

class RBOConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(RBOConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale_factor()
        self.Bilinear_binarization = Bilinear_binarization.apply
        self.out_channels = out_channels
        self.u = Parameter(0.2 * torch.ones(self.out_channels, 1, 1, 1))
        self.thre = 0.6
        
    def generate_scale_factor(self):
        self.scale_factor = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def recurrent_module(self, alpha, w):
        backtrack_varible = w.grad.clone()
        weight = w - self.u * self.drelu(alpha, w, backtrack_varible)
        return weight

    def drelu(self, alpha, w, backtrack_varible):
        _, idx = torch.sort(alpha, dim=0, descending=False, out=None)
        indicator = (torch.sign(idx.detach() - int(self.out_channels * (1 - self.thre)) + 0.5) - 1).detach()/ (-2)
        return backtrack_varible * indicator

    def forward(self, x):

        scale_factor = torch.abs(self.scale_factor)

        if (self.weight.grad is not None) and (self.training):
            weight = self.recurrent_module(scale_factor, self.weight)
        else:
            weight = self.weight

        new_weight = self.Bilinear_binarization(weight, scale_factor)

        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, param=1e-4):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = RBOConv(inplanes, planes, stride=stride, padding=1, bias=False, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.params=[1e-4, 1e-4, 1e-4, 1e-4]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], param=self.params[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, param=self.params[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, param=self.params[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, param=self.params[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, param = 1e-4):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
   
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, param=param))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, param=param))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        

def rbonn18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

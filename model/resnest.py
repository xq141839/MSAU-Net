##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""
 
import torch
from model import resnet
 
ResNet = resnet.ResNet
Bottleneck = resnet.Bottleneck
 
def resnest18(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=8, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model
 
 
 
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
import numpy as np
import torch.nn.functional as F
import math
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,TensorDataset
from os import listdir
import os
from PIL import Image
from torchvision import transforms
import scipy.io
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import random
# from Dataset import ciaData
import cv2 as cv #8.3添加
from kornia.color import lab_to_rgb,rgb_to_lab,rgb_to_hsv,hsv_to_rgb#8.28添加
from torchvision.hed import hed_to_rgb,rgb_to_hed#9.11
from hed import hed_to_rgb,rgb_to_hed#9.11
from norm4D import norm4D,unorm4D

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mean_ = (0.485, 0.456, 0.406)
std_ = (0.229, 0.224, 0.225)


mean_13= (58.77515069,27.25982541,-7.825135523)
std_13 =(20.70605765,12.67681833,6.666722412)

mean_2= (50.69584084,33.70910395,-9.417649872)
std_2= (14.18854242,7.071403321,6.476873549)

mean_6= (26.38038809,21.939772,-18.54567921)
std_6= (21.07028115,6.905576127,8.149537741)

mean_8= (52.40771309,15.71980628,-14.10899633)
std_8= (30.20962623,9.022655809,11.06458503)

mean_9= (69.79752839,18.39389349,-8.408362564)
std_9 =(19.91183487,11.46137616,7.233493937)


mean_lab = (51.61 ,23.40 ,-11.66 )
std_lab = (21.22 ,9.43,7.92 )


mean_hsv = (0.82702367,0.37666315,0.631354464)
std_hsv = (0.147702595,0.171121216,0.19210163)

mean_hed = (0.6411, -0.3809, 0.5712)
std_hed = (0.3239, 0.0990, 0.1606)



__all__ = [
    "ResNet",
    "ResNet13",
    "ResNet2",
    "ResNet6",
    "ResNet8",
    "ResNet9",
    "ResNet_mean",
    "ResNet_lab_norm_6",#9.11
    "ResNet_lab_norm_8",
    "ResNet_lab_norm_mean",
    "ResNet_hsv_norm",
    "ResNet_hed_norm",
    "ResNet_concat",
    "ResNet_avg",
    "ResNet_weighted_avg",
    
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet18_13",
    "resnet18_2",
    "resnet18_6",
    "resnet18_8",
    "resnet18_9",
    "resnet18_mean",
    "resnet18_lab_norm_6",#9.11
    "resnet18_lab_norm_8",
    "resnet18_lab_norm_mean",
    "resnet18_hsv_norm",
    "resnet18_hed_norm",
    "resnet18_concat",
    "resnet18_avg",
    "resnet18_weighted_avg",    
    "resnet34",    
    "resnet50",    
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


class bw_13(nn.Module):  #8.25
    def __init__(self):            
        super(bw_13, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_13,std_13) 

        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4
    
class bw_2(nn.Module):  #8.25
    def __init__(self):            
        super(bw_2, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_2,std_2) 
        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4
    
class bw_6(nn.Module):  #8.25
    def __init__(self):            
        super(bw_6, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_6,std_6) 
        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4
    
class bw_8(nn.Module):  #8.25
    def __init__(self):            
        super(bw_8, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_8,std_8) 
        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4

class bw_9(nn.Module):  #8.25
    def __init__(self):            
        super(bw_9, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_9,std_9) 
        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4
    
class bw_mean(nn.Module):  #8.25
    def __init__(self):            
        super(bw_mean, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=False,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        xxx = norm4D(output0,mean_lab,std_lab) 
        x4 = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
#         x4 = xxx   
        x4 = unorm4D(x4,mean_,std_) 
        return x4
    
class bw_lab_8(nn.Module):  #8.25
    def __init__(self):            
        super(bw_lab_8, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        x0 = output0[..., 0, :, :]
        x1 = output0[..., 1, :, :]
        x2 = output0[..., 2, :, :]
        x0 = x0*self.weight0[0]+self.bias0[0]
        x1 = x1*self.weight0[1]+self.bias0[1]
        x2 = x2*self.weight0[2]+self.bias0[2]
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_8,std_8)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
        xxx = torch.stack([x0,x1,x2], -3).to(device)
        xxx = norm4D(xxx,mean_8,std_8)
        xxx = lab_to_rgb(xxx)
        print(self.weight0, self.bias0)
        x4 = xxx
        x4 = unorm4D(xxx,mean_,std_) 
        return x4
    
class bw_lab_6(nn.Module):  #8.25
    def __init__(self):            
        super(bw_lab_6, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        x0 = output0[..., 0, :, :]
        x1 = output0[..., 1, :, :]
        x2 = output0[..., 2, :, :]
        x0 = x0*self.weight0[0]+self.bias0[0]
        x1 = x1*self.weight0[1]+self.bias0[1]
        x2 = x2*self.weight0[2]+self.bias0[2]
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_8,std_8)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
        xxx = torch.stack([x0,x1,x2], -3).to(device)
        xxx = norm4D(xxx,mean_6,std_6)
        xxx = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
        x4 = xxx
        x4 = unorm4D(xxx,mean_,std_) 
        return x4
    
# class bw_lab_6(nn.Module):  #8.25
#     def __init__(self):            
#         super(bw_lab_6, self).__init__()
#         self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
#     def forward(self, x):
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx = rgb_to_lab(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output0 = n(xx)
#         x0 = output0[..., 0, :, :]
#         x1 = output0[..., 1, :, :]
#         x2 = output0[..., 2, :, :]
#         x0 = x0*(0.1*self.weight0[0] + 1)
#         x1 = x1*(0.1*self.weight0[1] + 1)
#         x2 = x2*(0.1*self.weight0[2] + 1)
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_6,std_6)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
#         xxx = torch.stack([y0,y1,y2], -3).to(device)
#         xxx = lab_to_rgb(xxx)
# #         print(self.weight0, self.bias0)
#         x4 = xxx
#         x4 = unorm4D(xxx,mean_,std_) 
#         return x4

class bw_lab(nn.Module):  #8.25
    def __init__(self):            
        super(bw_lab, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        x0 = output0[..., 0, :, :]
        x1 = output0[..., 1, :, :]
        x2 = output0[..., 2, :, :]
        x0 = x0*self.weight0[0]+self.bias0[0]
        x1 = x1*self.weight0[1]+self.bias0[1]
        x2 = x2*self.weight0[2]+self.bias0[2]
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_8,std_8)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
        xxx = torch.stack([x0,x1,x2], -3).to(device)
        xxx = norm4D(xxx,mean_lab,std_lab)
        xxx = lab_to_rgb(xxx)
#         print(self.weight0, self.bias0)
        x4 = xxx
        x4 = unorm4D(xxx,mean_,std_) 
        return x4
# class bw_lab(nn.Module):  #8.25
#     def __init__(self):            
#         super(bw_lab, self).__init__()
#         self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
#     def forward(self, x):
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx = rgb_to_lab(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output0 = n(xx)
#         x0 = output0[..., 0, :, :]
#         x1 = output0[..., 1, :, :]
#         x2 = output0[..., 2, :, :]
#         x0 = x0*(0.1*self.weight0[0] + 1)
#         x1 = x1*(0.1*self.weight0[1] + 1)
#         x2 = x2*(0.1*self.weight0[2] + 1)
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_lab,std_lab)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
#         xxx = torch.stack([y0,y1,y2], -3).to(device)
#         xxx = lab_to_rgb(xxx)
# #         print(self.weight0, self.bias0)
#         x4 = xxx
#         x4 = unorm4D(xxx,mean_,std_) 
#         return x4

class bw_hsv(nn.Module):  #8.25
    def __init__(self):            
        super(bw_hsv, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_hsv(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        x0 = output0[..., 0, :, :]
        x1 = output0[..., 1, :, :]
        x2 = output0[..., 2, :, :]
        x0 = x0*self.weight0[0]+self.bias0[0]
        x1 = x1*self.weight0[1]+self.bias0[1]
        x2 = x2*self.weight0[2]+self.bias0[2]
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_8,std_8)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
        xxx = torch.stack([x0,x1,x2], -3).to(device)
        xxx = norm4D(xxx,mean_hsv,std_hsv)
        xxx = hsv_to_rgb(xxx)
#         print(self.weight0, self.bias0)
        x4 = xxx
        x4 = unorm4D(xxx,mean_,std_) 
        return x4
# class bw_hsv(nn.Module):  #8.25
#     def __init__(self):            
#         super(bw_hsv, self).__init__()
#         self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
#     def forward(self, x):
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx = rgb_to_hsv(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output0 = n(xx)
#         x0 = output0[..., 0, :, :]
#         x1 = output0[..., 1, :, :]
#         x2 = output0[..., 2, :, :]
#         x0 = x0*(0.1*self.weight0[0] + 1)
#         x1 = x1*(0.1*self.weight0[1] + 1)
#         x2 = x2*(0.1*self.weight0[2] + 1)
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_hsv,std_hsv)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
#         xxx = torch.stack([y0,y1,y2], -3).to(device)
#         xxx = hsv_to_rgb(xxx)
# #         print(self.weight0, self.bias0)
#         x4 = xxx
#         x4 = unorm4D(xxx,mean_,std_) 
#         return x4

class bw_hed(nn.Module):  #8.25
    def __init__(self):            
        super(bw_hed, self).__init__()
        self.weight0=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx = rgb_to_hed(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        x0 = output0[..., 0, :, :]
        x1 = output0[..., 1, :, :]
        x2 = output0[..., 2, :, :]
        x0 = x0*self.weight0[0]+self.bias0[0]
        x1 = x1*self.weight0[1]+self.bias0[1]
        x2 = x2*self.weight0[2]+self.bias0[2]
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_8,std_8)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
        xxx = torch.stack([x0,x1,x2], -3).to(device)
        xxx = norm4D(xxx,mean_hed,std_hed)
        xxx = hed_to_rgb(xxx)
#         print(self.weight0, self.bias0)
        x4 = xxx
        x4 = unorm4D(xxx,mean_,std_) 
        return x4
    
# class bw_hed(nn.Module):  #8.25
#     def __init__(self):            
#         super(bw_hed, self).__init__()
#         self.weight0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias0=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
#     def forward(self, x):
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx = rgb_to_hed(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output0 = n(xx)
#         x0 = output0[..., 0, :, :]
#         x1 = output0[..., 1, :, :]
#         x2 = output0[..., 2, :, :]
#         x0 = x0*(0.1*self.weight0[0] + 1)
#         x1 = x1*(0.1*self.weight0[1] + 1)
#         x2 = x2*(0.1*self.weight0[2] + 1)
#         output = torch.stack([x0,x1,x2], -3).to(device)
#         output = norm4D(output,mean_hed,std_hed)
#         y0 = output[..., 0, :, :]+self.bias0[0]
#         y1 = output[..., 1, :, :]+self.bias0[1]
#         y2 = output[..., 2, :, :]+self.bias0[2]
#         xxx = torch.stack([y0,y1,y2], -3).to(device)
#         xxx = hed_to_rgb(xxx)
# #         print(self.weight0, self.bias0)
#         x4 = xxx
#         x4 = unorm4D(xxx,mean_,std_) 
#         return x4
    
    
#9.22    
class bw_concat(nn.Module):  #8.25
    def __init__(self):            
        super(bw_concat, self).__init__()
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx_lab = rgb_to_lab(xun).to(device)
        xx_hsv = rgb_to_hsv(xun).to(device)
        xx_hed = rgb_to_hed(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output_lab = n(xx_lab)
        output_hsv = n(xx_hsv)
        output_hed = n(xx_hed)
        
        lab0 = output_lab[..., 0, :, :]
        lab1 = output_lab[..., 1, :, :]
        lab2 = output_lab[..., 2, :, :]
        hsv0 = output_hsv[..., 0, :, :]
        hsv1 = output_hsv[..., 1, :, :]
        hsv2 = output_hsv[..., 2, :, :]
        hed0 = output_hed[..., 0, :, :]
        hed1 = output_hed[..., 1, :, :]
        hed2 = output_hed[..., 2, :, :]
        
        lab0 = lab0*self.weight_lab[0]+self.bias_lab[0]
        lab1 = lab1*self.weight_lab[1]+self.bias_lab[1]
        lab2 = lab2*self.weight_lab[2]+self.bias_lab[2]
        hsv0 = hsv0*self.weight_hsv[0]+self.bias_hsv[0]
        hsv1 = hsv1*self.weight_hsv[1]+self.bias_hsv[1]
        hsv2 = hsv2*self.weight_hsv[2]+self.bias_hsv[2]
        hed0 = hed0*self.weight_hed[0]+self.bias_hed[0]
        hed1 = hed1*self.weight_hed[1]+self.bias_hed[1]
        hed2 = hed2*self.weight_hed[2]+self.bias_hed[2]
        
        
#         lab0 = lab0*(0.1*self.weight_lab[0] + 1)
#         lab1 = lab1*(0.1*self.weight_lab[1] + 1)
#         lab2 = lab2*(0.1*self.weight_lab[2] + 1)
#         hsv0 = hsv0*(0.1*self.weight_hsv[0] + 1)
#         hsv1 = hsv1*(0.1*self.weight_hsv[1] + 1)
#         hsv2 = hsv2*(0.1*self.weight_hsv[2] + 1)
#         hed0 = hed0*(0.1*self.weight_hed[0] + 1)
#         hed1 = hed1*(0.1*self.weight_hed[1] + 1)
#         hed2 = hed2*(0.1*self.weight_hed[2] + 1)
        
        output_lab = torch.stack([lab0,lab1,lab2], -3).to(device)
        output_hsv = torch.stack([hsv0,hsv1,hsv2], -3).to(device)
        output_hed = torch.stack([hed0,hed1,hed2], -3).to(device)
        
        
        output_lab = norm4D(output_lab,mean_lab,std_lab)
        output_hsv = norm4D(output_hsv,mean_hsv,std_hsv)
        output_hed = norm4D(output_hed,mean_hed,std_hed)
        
#         laby0 = output_lab[..., 0, :, :]+self.bias_lab[0]
#         laby1 = output_lab[..., 1, :, :]+self.bias_lab[1]
#         laby2 = output_lab[..., 2, :, :]+self.bias_lab[2]
#         hsvy0 = output_hsv[..., 0, :, :]+self.bias_hsv[0]
#         hsvy1 = output_hsv[..., 1, :, :]+self.bias_hsv[1]
#         hsvy2 = output_hsv[..., 2, :, :]+self.bias_hsv[2]
#         hedy0 = output_hed[..., 0, :, :]+self.bias_hed[0]
#         hedy1 = output_hed[..., 1, :, :]+self.bias_hed[1]
#         hedy2 = output_hed[..., 2, :, :]+self.bias_hed[2]
        
#         xxx_lab = torch.stack([laby0,laby1,laby2], -3).to(device)
#         xxx_hsv = torch.stack([hsvy0,hsvy1,hsvy2], -3).to(device)
#         xxx_hed = torch.stack([hedy0,hedy1,hedy2], -3).to(device)
        
        xxx_lab = lab_to_rgb(output_lab)
        xxx_hsv = hsv_to_rgb(output_hsv)
        xxx_hed = hed_to_rgb(output_hed)
        
#         print(self.weight0, self.bias0)
        x4_lab = unorm4D(xxx_lab,mean_,std_) 
        x4_hsv = unorm4D(xxx_hsv,mean_,std_) 
        x4_hed = unorm4D(xxx_hed,mean_,std_) 
        x4 = torch.cat([x4_lab,x4_hsv,x4_hed],1)
        return x4
    
class bw_avg(nn.Module):  #8.25
    def __init__(self):            
        super(bw_concat, self).__init__()
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx_lab = rgb_to_lab(xun).to(device)
        xx_hsv = rgb_to_hsv(xun).to(device)
        xx_hed = rgb_to_hed(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output_lab = n(xx_lab)
        output_hsv = n(xx_hsv)
        output_hed = n(xx_hed)
        
        lab0 = output_lab[..., 0, :, :]
        lab1 = output_lab[..., 1, :, :]
        lab2 = output_lab[..., 2, :, :]
        hsv0 = output_hsv[..., 0, :, :]
        hsv1 = output_hsv[..., 1, :, :]
        hsv2 = output_hsv[..., 2, :, :]
        hed0 = output_hed[..., 0, :, :]
        hed1 = output_hed[..., 1, :, :]
        hed2 = output_hed[..., 2, :, :]
        
        lab0 = lab0*self.weight_lab[0]+self.bias_lab[0]
        lab1 = lab1*self.weight_lab[1]+self.bias_lab[1]
        lab2 = lab2*self.weight_lab[2]+self.bias_lab[2]
        hsv0 = hsv0*self.weight_hsv[0]+self.bias_hsv[0]
        hsv1 = hsv1*self.weight_hsv[1]+self.bias_hsv[1]
        hsv2 = hsv2*self.weight_hsv[2]+self.bias_hsv[2]
        hed0 = hed0*self.weight_hed[0]+self.bias_hed[0]
        hed1 = hed1*self.weight_hed[1]+self.bias_hed[1]
        hed2 = hed2*self.weight_hed[2]+self.bias_hed[2]
        
        output_lab = torch.stack([lab0,lab1,lab2], -3).to(device)
        output_hsv = torch.stack([hsv0,hsv1,hsv2], -3).to(device)
        output_hed = torch.stack([hed0,hed1,hed2], -3).to(device)
        
        
        output_lab = norm4D(output_lab,mean_lab,std_lab)
        output_hsv = norm4D(output_hsv,mean_hsv,std_hsv)
        output_hed = norm4D(output_hed,mean_hed,std_hed)
        
        xxx_lab = lab_to_rgb(output_lab)
        xxx_hsv = hsv_to_rgb(output_hsv)
        xxx_hed = hed_to_rgb(output_hed)
        
#         print(self.weight0, self.bias0)
        x4_lab = unorm4D(xxx_lab,mean_,std_) 
        x4_hsv = unorm4D(xxx_hsv,mean_,std_) 
        x4_hed = unorm4D(xxx_hed,mean_,std_) 
        x4 = (x4_lab + x4_hsv + x4_hed)/3
        return x4
    
# class bw_avg(nn.Module):  #8.25
#     def __init__(self):            
#         super(bw_avg, self).__init__()
#         self.weight_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.weight_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.weight_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
#         self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
#     def forward(self, x):
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx_lab = rgb_to_lab(xun).to(device)
#         xx_hsv = rgb_to_hsv(xun).to(device)
#         xx_hed = rgb_to_hed(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output_lab = n(xx_lab)
#         output_hsv = n(xx_hsv)
#         output_hed = n(xx_hed)
        
#         lab0 = output_lab[..., 0, :, :]
#         lab1 = output_lab[..., 1, :, :]
#         lab2 = output_lab[..., 2, :, :]
#         hsv0 = output_hsv[..., 0, :, :]
#         hsv1 = output_hsv[..., 1, :, :]
#         hsv2 = output_hsv[..., 2, :, :]
#         hed0 = output_hed[..., 0, :, :]
#         hed1 = output_hed[..., 1, :, :]
#         hed2 = output_hed[..., 2, :, :]
        
        
#         lab0 = lab0*(0.1*self.weight_lab[0] + 1)
#         lab1 = lab1*(0.1*self.weight_lab[1] + 1)
#         lab2 = lab2*(0.1*self.weight_lab[2] + 1)
#         hsv0 = hsv0*(0.1*self.weight_hsv[0] + 1)
#         hsv1 = hsv1*(0.1*self.weight_hsv[1] + 1)
#         hsv2 = hsv2*(0.1*self.weight_hsv[2] + 1)
#         hed0 = hed0*(0.1*self.weight_hed[0] + 1)
#         hed1 = hed1*(0.1*self.weight_hed[1] + 1)
#         hed2 = hed2*(0.1*self.weight_hed[2] + 1)
        
#         output_lab = torch.stack([lab0,lab1,lab2], -3).to(device)
#         output_hsv = torch.stack([hsv0,hsv1,hsv2], -3).to(device)
#         output_hed = torch.stack([hed0,hed1,hed2], -3).to(device)
        
        
#         output_lab = norm4D(output_lab,mean_lab,std_lab)
#         output_hsv = norm4D(output_hsv,mean_hsv,std_hsv)
#         output_hed = norm4D(output_hed,mean_hed,std_hed)
        
#         laby0 = output_lab[..., 0, :, :]+self.bias_lab[0]
#         laby1 = output_lab[..., 1, :, :]+self.bias_lab[1]
#         laby2 = output_lab[..., 2, :, :]+self.bias_lab[2]
#         hsvy0 = output_hsv[..., 0, :, :]+self.bias_hsv[0]
#         hsvy1 = output_hsv[..., 1, :, :]+self.bias_hsv[1]
#         hsvy2 = output_hsv[..., 2, :, :]+self.bias_hsv[2]
#         hedy0 = output_hed[..., 0, :, :]+self.bias_hed[0]
#         hedy1 = output_hed[..., 1, :, :]+self.bias_hed[1]
#         hedy2 = output_hed[..., 2, :, :]+self.bias_hed[2]
        
#         xxx_lab = torch.stack([laby0,laby1,laby2], -3).to(device)
#         xxx_hsv = torch.stack([hsvy0,hsvy1,hsvy2], -3).to(device)
#         xxx_hed = torch.stack([hedy0,hedy1,hedy2], -3).to(device)
        
#         xxx_lab = lab_to_rgb(xxx_lab)
#         xxx_hsv = hsv_to_rgb(xxx_hsv)
#         xxx_hed = hed_to_rgb(xxx_hed)
        
# #         print(self.weight0, self.bias0)
#         x4_lab = unorm4D(xxx_lab,mean_,std_) 
#         x4_hsv = unorm4D(xxx_hsv,mean_,std_) 
#         x4_hed = unorm4D(xxx_hed,mean_,std_) 
#         x4 = (x4_lab + x4_hsv + x4_hed)/3
#         return x4
    
    
class bw_weighted_avg(nn.Module):  #8.25
    def __init__(self):            
        super(bw_weighted_avg, self).__init__()
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.w1=nn.Parameter(torch.tensor([0.33,0.33,0.33], dtype=torch.float32, requires_grad=True,device=device))
        self.w2=nn.Parameter(torch.tensor([0.33,0.33,0.33], dtype=torch.float32, requires_grad=True,device=device))              
        
    def forward(self, x):
        xun = x
        xun = norm4D(xun,mean_,std_)
        xx_lab = rgb_to_lab(xun).to(device)
        xx_hsv = rgb_to_hsv(xun).to(device)
        xx_hed = rgb_to_hed(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output_lab = n(xx_lab)
        output_hsv = n(xx_hsv)
        output_hed = n(xx_hed)
        
        lab0 = output_lab[..., 0, :, :]
        lab1 = output_lab[..., 1, :, :]
        lab2 = output_lab[..., 2, :, :]
        hsv0 = output_hsv[..., 0, :, :]
        hsv1 = output_hsv[..., 1, :, :]
        hsv2 = output_hsv[..., 2, :, :]
        hed0 = output_hed[..., 0, :, :]
        hed1 = output_hed[..., 1, :, :]
        hed2 = output_hed[..., 2, :, :]
        
        lab0 = lab0*self.weight_lab[0]+self.bias_lab[0]
        lab1 = lab1*self.weight_lab[1]+self.bias_lab[1]
        lab2 = lab2*self.weight_lab[2]+self.bias_lab[2]
        hsv0 = hsv0*self.weight_hsv[0]+self.bias_hsv[0]
        hsv1 = hsv1*self.weight_hsv[1]+self.bias_hsv[1]
        hsv2 = hsv2*self.weight_hsv[2]+self.bias_hsv[2]
        hed0 = hed0*self.weight_hed[0]+self.bias_hed[0]
        hed1 = hed1*self.weight_hed[1]+self.bias_hed[1]
        hed2 = hed2*self.weight_hed[2]+self.bias_hed[2]
        
        output_lab = torch.stack([lab0,lab1,lab2], -3).to(device)
        output_hsv = torch.stack([hsv0,hsv1,hsv2], -3).to(device)
        output_hed = torch.stack([hed0,hed1,hed2], -3).to(device)
        
        
        output_lab = norm4D(output_lab,mean_lab,std_lab)
        output_hsv = norm4D(output_hsv,mean_hsv,std_hsv)
        output_hed = norm4D(output_hed,mean_hed,std_hed)
        
        xxx_lab = lab_to_rgb(output_lab)
        xxx_hsv = hsv_to_rgb(output_hsv)
        xxx_hed = hed_to_rgb(output_hed)
        
#         print(self.weight0, self.bias0)
        x4_lab = unorm4D(xxx_lab,mean_,std_) 
        x4_hsv = unorm4D(xxx_hsv,mean_,std_) 
        x4_hed = unorm4D(xxx_hed,mean_,std_) 
        x4_lab0 = x4_lab[..., 0, :, :]*self.w1[0]
        x4_lab1 = x4_lab[..., 1, :, :]*self.w1[1]
        x4_lab2 = x4_lab[..., 2, :, :]*self.w1[2]
        x4_hsv0 = x4_hsv[..., 0, :, :]*self.w2[0]
        x4_hsv1 = x4_hsv[..., 1, :, :]*self.w2[1]
        x4_hsv2 = x4_hsv[..., 2, :, :]*self.w2[2]
        x4_hed0 = x4_hed[..., 0, :, :]*(1-self.w1[0]-self.w2[0])
        x4_hed1 = x4_hed[..., 1, :, :]*(1-self.w1[1]-self.w2[1])
        x4_hed2 = x4_hed[..., 2, :, :]*(1-self.w1[2]-self.w2[2])
        
        x5_lab = torch.stack([x4_lab0,x4_lab1,x4_lab2], -3).to(device)
        x5_hsv = torch.stack([x4_hsv0,x4_hsv1,x4_hsv2], -3).to(device)
        x5_hed = torch.stack([x4_hed0,x4_hed1,x4_hed2], -3).to(device)
        x5 = x5_lab + x5_hsv + x5_hed
        return x5
#         xun = x
#         xun = norm4D(xun,mean_,std_)
#         xx_lab = rgb_to_lab(xun).to(device)
#         xx_hsv = rgb_to_hsv(xun).to(device)
#         xx_hed = rgb_to_hed(xun).to(device)
#         m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
#         n = m.to(device)
#         output_lab = n(xx_lab)
#         output_hsv = n(xx_hsv)
#         output_hed = n(xx_hed)
        
#         lab0 = output_lab[..., 0, :, :]
#         lab1 = output_lab[..., 1, :, :]
#         lab2 = output_lab[..., 2, :, :]
#         hsv0 = output_hsv[..., 0, :, :]
#         hsv1 = output_hsv[..., 1, :, :]
#         hsv2 = output_hsv[..., 2, :, :]
#         hed0 = output_hed[..., 0, :, :]
#         hed1 = output_hed[..., 1, :, :]
#         hed2 = output_hed[..., 2, :, :]
        
        
#         lab0 = lab0*(0.1*self.weight_lab[0] + 1)
#         lab1 = lab1*(0.1*self.weight_lab[1] + 1)
#         lab2 = lab2*(0.1*self.weight_lab[2] + 1)
#         hsv0 = hsv0*(0.1*self.weight_hsv[0] + 1)
#         hsv1 = hsv1*(0.1*self.weight_hsv[1] + 1)
#         hsv2 = hsv2*(0.1*self.weight_hsv[2] + 1)
#         hed0 = hed0*(0.1*self.weight_hed[0] + 1)
#         hed1 = hed1*(0.1*self.weight_hed[1] + 1)
#         hed2 = hed2*(0.1*self.weight_hed[2] + 1)
        
#         output_lab = torch.stack([lab0,lab1,lab2], -3).to(device)
#         output_hsv = torch.stack([hsv0,hsv1,hsv2], -3).to(device)
#         output_hed = torch.stack([hed0,hed1,hed2], -3).to(device)
        
        
#         output_lab = norm4D(output_lab,mean_lab,std_lab)
#         output_hsv = norm4D(output_hsv,mean_hsv,std_hsv)
#         output_hed = norm4D(output_hed,mean_hed,std_hed)
        
#         laby0 = output_lab[..., 0, :, :]+self.bias_lab[0]
#         laby1 = output_lab[..., 1, :, :]+self.bias_lab[1]
#         laby2 = output_lab[..., 2, :, :]+self.bias_lab[2]
#         hsvy0 = output_hsv[..., 0, :, :]+self.bias_hsv[0]
#         hsvy1 = output_hsv[..., 1, :, :]+self.bias_hsv[1]
#         hsvy2 = output_hsv[..., 2, :, :]+self.bias_hsv[2]
#         hedy0 = output_hed[..., 0, :, :]+self.bias_hed[0]
#         hedy1 = output_hed[..., 1, :, :]+self.bias_hed[1]
#         hedy2 = output_hed[..., 2, :, :]+self.bias_hed[2]
        
#         xxx_lab = torch.stack([laby0,laby1,laby2], -3).to(device)
#         xxx_hsv = torch.stack([hsvy0,hsvy1,hsvy2], -3).to(device)
#         xxx_hed = torch.stack([hedy0,hedy1,hedy2], -3).to(device)
        
#         xxx_lab = lab_to_rgb(xxx_lab)
#         xxx_hsv = hsv_to_rgb(xxx_hsv)
#         xxx_hed = hed_to_rgb(xxx_hed)
        
# #         print(self.weight0, self.bias0)
#         x4_lab = unorm4D(xxx_lab,mean_,std_)
#         x4_hsv = unorm4D(xxx_hsv,mean_,std_) 
#         x4_hed = unorm4D(xxx_hed,mean_,std_) 
        
#         x4_lab0 = x4_lab[..., 0, :, :]*self.w1[0]
#         x4_lab1 = x4_lab[..., 1, :, :]*self.w1[1]
#         x4_lab2 = x4_lab[..., 2, :, :]*self.w1[2]
#         x4_hsv0 = x4_hsv[..., 0, :, :]*self.w2[0]
#         x4_hsv1 = x4_hsv[..., 1, :, :]*self.w2[1]
#         x4_hsv2 = x4_hsv[..., 2, :, :]*self.w2[2]
#         x4_hed0 = x4_hed[..., 0, :, :]*(1-self.w1[0]-self.w2[0])
#         x4_hed1 = x4_hed[..., 1, :, :]*(1-self.w1[1]-self.w2[1])
#         x4_hed2 = x4_hed[..., 2, :, :]*(1-self.w1[2]-self.w2[2])
        
#         x5_lab = torch.stack([x4_lab0,x4_lab1,x4_lab2], -3).to(device)
#         x5_hsv = torch.stack([x4_hsv0,x4_hsv1,x4_hsv2], -3).to(device)
#         x5_hed = torch.stack([x4_hed0,x4_hed1,x4_hed2], -3).to(device)
#         x5 = x5_lab + x5_hsv + x5_hed
#         return x5
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet13(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_13()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet2(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_2()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet6(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_6()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class ResNet8(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_8()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class ResNet9(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_9()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class ResNet_mean(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_mean()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet_lab_norm_8(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_lab_8()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet_lab_norm_6(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_lab_6()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet_lab_norm_mean(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_lab()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNet_hsv_norm(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_hsv()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNet_hed_norm(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_hed()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    
#9.22
class ResNet_concat(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_concat()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(9, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)#9.22
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
class ResNet_avg(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_avg()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class ResNet_weighted_avg(nn.Module):#9.11
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
    ) -> None:
        super().__init__()
        self.bw = bw_weighted_avg()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bw(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet6(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet6:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet6(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet9(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet9:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet9(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet2(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet2:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet2(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet13(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet13:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet13(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet8(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet8:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet8(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_mean(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_mean:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_mean(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_lab_norm_6(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_lab_norm_6:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_lab_norm_6(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_lab_norm_8(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_lab_norm_8:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_lab_norm_8(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_lab_norm(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_lab_norm_mean:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_lab_norm_mean(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_hsv_norm(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_hsv_norm:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_hsv_norm(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_hed_norm(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_hed_norm:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_hed_norm(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

#9.22
def _resnet_concat(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_concat:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_concat(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_avg(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_avg:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_avg(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def _resnet_weighted_avg(#9.11
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_weighted_avg:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_weighted_avg(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet34_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 21797672,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet101_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet152_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 83455272,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }
            },
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class Wide_ResNet101_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_6(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet6:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet6(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_9(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet9:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet9(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_2(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet2:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet2(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_13(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet13:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet13(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_8(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet8:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet8(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_mean(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_mean:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_mean(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet18_lab_norm_8(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_lab_norm_8:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_lab_norm_8(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
def resnet18_lab_norm_6(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_lab_norm_6:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_lab_norm_6(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_lab_norm_mean(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_lab_norm_mean:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_lab_norm(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_hsv_norm(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_hsv_norm:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_hsv_norm(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_hed_norm(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_hed_norm:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_hed_norm(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

#9.22
def resnet18_concat(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_concat:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_concat(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_avg(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_avg:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_avg(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet18_weighted_avg(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_weighted_avg:#9.11
    weights = ResNet18_Weights.verify(weights)
    return _resnet_weighted_avg(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)



@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# The dictionary below is internal implementation detail and will be removed in v0.15
from ._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
        "resnet34": ResNet34_Weights.IMAGENET1K_V1.url,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1.url,
        "resnet101": ResNet101_Weights.IMAGENET1K_V1.url,
        "resnet152": ResNet152_Weights.IMAGENET1K_V1.url,
        "resnext50_32x4d": ResNeXt50_32X4D_Weights.IMAGENET1K_V1.url,
        "resnext101_32x8d": ResNeXt101_32X8D_Weights.IMAGENET1K_V1.url,
        "wide_resnet50_2": Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
        "wide_resnet101_2": Wide_ResNet101_2_Weights.IMAGENET1K_V1.url,
    }
)

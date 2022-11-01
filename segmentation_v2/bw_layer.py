import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from os import listdir
import os
from PIL import Image
from torchvision import transforms
from bw_layer import *
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
# from Dataset import ciaData
import cv2 as cv  # 8.3添加
from kornia.color import lab_to_rgb,rgb_to_lab

img = Image.open(r'/root/autodl-tmp/RandStainNA-master/segmentation/choose25/TCGA-18-5592-01Z-00-DX1_0_512.png').convert('RGB')
transform1 = transforms.Compose([transforms.ToTensor()])
img0 = transform1(img)
img1 = img0.unsqueeze(0)
# print(img1)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mean_ = (0.485, 0.456, 0.406)
std_ = (0.229, 0.224, 0.225)
mean_0 = (59, 31, -5.6)  # 8.8添加lab_norm初始值，依据是template中表现最好的一张，之后可用数据集平均值
std_0 = (22, 14, 8.5)


# 8.2反归一化
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            with torch.no_grad():  # 8.7添加
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                t.sub_(m).div_(s)
        return tensor

class bw(nn.Module):  # 8.25
    def __init__(self):
        super(bw, self).__init__()
        self.weight0 = nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True, device=device))
        self.bias0 = nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True, device=device))
#     def rgb2lab(self, x):
#         input0 = x
#         input1 = x
#         for i in range(0, x.shape[0]):
#             unorm = UnNormalize(mean=mean_, std=std_)  # 8.2反归一化
#             #                     print(input0[i])
#             input1[i] = unorm(input0[i])
#             #                     input0[i] = 255*input0[i]  #8.15debug 范围0-1否则转lab失效
#             input0_ = input1[i].permute(1, 2, 0)
#             input0_n = input0_.data.cpu().numpy()
#             input0_lab = cv.cvtColor(input0_n, cv.COLOR_RGB2LAB)
#             input0_t = torch.tensor(input0_lab, device=device).float()
#             input0[i] = input0_t.permute(2, 0, 1)
#         return input0
    def forward(self, x):
        xun = x
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                xun[i][j] = x[i][j].to(device) * std_[j] + mean_[j]
        xx = rgb_to_lab(xun).to(device)
        m = nn.InstanceNorm2d(3, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        n = m.to(device)
        output0 = n(xx)
        output = output0.tolist()
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                output[i][j] = (x[i][j].to(device) * self.weight0[j] + self.bias0[j])*std_0[j]+mean_0[j]#list
            output[i] = torch.stack(output[i])
        output = torch.stack(output)
        # print(output)
        xxx = lab_to_rgb(output)
        x4 = xxx
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                x4[i][j] = (xxx[i][j].to(device) -mean_[j])/std_[j]
        
        
        return x4


class CIAnet_bw(nn.Module):
    def __init__(self):
        super(CIAnet_bw, self).__init__()
        self.bw = bw()

    def forward(self, x):
        output = self.bw(x)# 8.25
        output_ = output[0][0][0][0]
        print(output_)
        # output = torch.tensor(np.mean(output))
        return (output_)


model = CIAnet_bw()
b = model(img1)
# print('version',b._version)
b.backward()
for parameters in model.parameters():
    print(parameters.grad)
